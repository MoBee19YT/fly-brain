package eu.flybrain.flybee;

import com.mojang.brigadier.context.CommandContext;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.particles.ParticleTypes;
import net.minecraft.network.chat.Component;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.BlockTags;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.entity.EntityTypes;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.MoverType;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Fly Bee: a real fruit-fly brain (running in bee_server.py on this PC) lives in a Minecraft bee.
 *
 *  Eyes   - every 2 ticks the bee looks around through two compound eyes (2 x 16 x 24 view rays)
 *  Brain  - what it sees goes to the whole FlyWire brain; its steering neurons (DNa01/DNa02) answer
 *  Body   - the bee turns and flies as the brain says; a reflex keeps it ~2 blocks above the ground
 *  Treats - touching a flower = nectar (dopamine for the brain); bumping into walls = punishment
 */
public class FlyBee implements ModInitializer {
    static final int PORT = 5580;
    static final int EH = 16, EW = 24;          // rays per eye: rows (elevation) x columns (azimuth)
    static final double VIEW_DIST = 24.0;

    static Mob bee = null;
    static ServerLevel beeLevel = null;
    static Vec3 home = null;
    static long tick = 0;
    static int nectarPending = 0, bumpPending = 0, flowersTotal = 0;
    static final Map<BlockPos, Long> emptyFlowers = new HashMap<>();   // flower -> game tick it refills

    // brain link (runs on its own thread so the game never waits)
    static final AtomicReference<String> latestObs = new AtomicReference<>(null);
    static volatile float turn = 0f, speed = 0.5f;
    static volatile boolean dash = false, connected = false;

    @Override
    public void onInitialize() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(Commands.literal("flybee")
                .then(Commands.literal("spawn").executes(FlyBee::spawn))
                .then(Commands.literal("remove").executes(FlyBee::remove))
                .then(Commands.literal("status").executes(FlyBee::status))));
        ServerTickEvents.END_SERVER_TICK.register(FlyBee::onTick);
        Thread t = new Thread(FlyBee::brainLink, "FlyBee brain link");
        t.setDaemon(true);
        t.start();
    }

    // ------------------------------------------------------------------ commands
    static int spawn(CommandContext<CommandSourceStack> ctx) {
        CommandSourceStack src = ctx.getSource();
        ServerLevel level = src.getLevel();
        if (bee != null && bee.isAlive()) bee.discard();
        Vec3 p = src.getPosition().add(0, 1.5, 0);
        Mob b = EntityTypes.BEE.spawn(level, BlockPos.containing(p), EntitySpawnReason.COMMAND);
        if (b == null) return 0;
        b.setPos(p.x, p.y, p.z);
        b.setNoAi(true);            // the fly brain flies it, not Minecraft's bee AI
        b.setNoGravity(true);
        b.setPersistenceRequired();
        b.setCustomName(Component.literal("Fly"));
        b.setCustomNameVisible(true);
        bee = b; beeLevel = level; home = p; flowersTotal = 0;
        src.sendSuccess(() -> Component.literal("The fly brain now lives in this bee. "
            + (connected ? "Brain connected." : "Start bee_server.py on your PC so it can think!")), false);
        return 1;
    }

    static int remove(CommandContext<CommandSourceStack> ctx) {
        if (bee != null) bee.discard();
        bee = null;
        ctx.getSource().sendSuccess(() -> Component.literal("Bee removed."), false);
        return 1;
    }

    static int status(CommandContext<CommandSourceStack> ctx) {
        ctx.getSource().sendSuccess(() -> Component.literal("Fly bee: " + (bee != null && bee.isAlive() ? "alive" : "none")
            + ", brain " + (connected ? "connected" : "NOT connected (start bee_server.py)")
            + ", flowers found: " + flowersTotal), false);
        return 1;
    }

    // ------------------------------------------------------------------ body + senses, every tick
    static void onTick(MinecraftServer server) {
        tick++;
        if (bee == null || !bee.isAlive() || beeLevel == null) return;
        ServerLevel level = beeLevel;

        // --- steering from the brain (+ a homing reflex so it can't fly out of loaded chunks)
        float yaw = bee.getYRot();
        float steer = turn;
        if (home != null) {
            Vec3 toHome = home.subtract(bee.position());
            double dist = Math.sqrt(toHome.x * toHome.x + toHome.z * toHome.z);
            if (dist > 40) {
                float homeYaw = (float) (Math.toDegrees(Math.atan2(-toHome.x, toHome.z)));
                float diff = wrap(homeYaw - yaw);
                steer += Math.signum(diff) * (float) Math.min(1.0, (dist - 40) / 10.0);
            }
        }
        yaw += Math.max(-1f, Math.min(1f, steer)) * 10f;
        double v = 0.08 + 0.22 * Math.max(0f, Math.min(1f, speed)) + (dash ? 0.3 : 0.0);
        double rad = Math.toRadians(yaw);
        double dx = -Math.sin(rad) * v, dz = Math.cos(rad) * v;

        // --- altitude reflex: hover about 2 blocks above whatever is below
        int ground = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, (int) Math.floor(bee.getX()), (int) Math.floor(bee.getZ()));
        double targetY = ground + 2.0;
        double dy = Math.max(-0.2, Math.min(0.2, (targetY - bee.getY()) * 0.15)) + (dash ? 0.15 : 0.0);

        bee.move(MoverType.SELF, new Vec3(dx, dy, dz));
        bee.setYRot(yaw); bee.setYHeadRot(yaw); bee.setYBodyRot(yaw); bee.setXRot(0f);
        if (bee.horizontalCollision) bumpPending++;

        // --- nectar: touching a flower that isn't empty
        BlockPos c = bee.blockPosition();
        for (BlockPos p : BlockPos.betweenClosed(c.offset(-1, -2, -1), c.offset(1, 1, 1))) {
            BlockState s = level.getBlockState(p);
            if (!isFlower(s)) continue;
            if (Vec3.atCenterOf(p).distanceTo(bee.position()) > 1.6) continue;
            Long refill = emptyFlowers.get(p);
            if (refill != null && refill > tick) continue;
            emptyFlowers.put(p.immutable(), tick + 600);    // flower refills after 30 s
            nectarPending++; flowersTotal++;
            level.sendParticles(ParticleTypes.HAPPY_VILLAGER, p.getX() + 0.5, p.getY() + 0.7, p.getZ() + 0.5, 10, 0.3, 0.3, 0.3, 0.0);
        }

        // --- eyes: every 2 ticks (10 Hz), look around and hand the picture to the brain thread
        if (tick % 2 == 0) {
            byte[] img = look(level, yaw);
            String obs = String.format(java.util.Locale.ROOT, "B %d %.2f %.2f %.2f %.1f %d %d %s\n",
                tick, bee.getX(), bee.getY(), bee.getZ(), yaw, nectarPending, bumpPending,
                Base64.getEncoder().encodeToString(img));
            nectarPending = 0; bumpPending = 0;
            latestObs.set(obs);
        }
    }

    static boolean isFlower(BlockState s) {
        return s.is(BlockTags.FLOWERS) || s.is(BlockTags.SMALL_FLOWERS);
    }

    /** Two compound eyes: left eye looks from 175 deg left to 15 deg right, right eye mirrored. 4 bytes per ray: R,G,B,UV. */
    static byte[] look(ServerLevel level, float yaw) {
        byte[] out = new byte[2 * EH * EW * 4];
        Vec3 eye = bee.getEyePosition();
        int k = 0;
        for (int e = 0; e < 2; e++) {
            for (int r = 0; r < EH; r++) {
                float elev = 60f - 110f * r / (EH - 1);              // +60 (up) .. -50 (down)
                for (int col = 0; col < EW; col++) {
                    float az = e == 0 ? (-175f + 190f * col / (EW - 1)) : (-15f + 190f * col / (EW - 1));
                    Vec3 dir = Vec3.directionFromRotation(-elev, yaw + az);
                    Vec3 end = eye.add(dir.scale(VIEW_DIST));
                    BlockHitResult hit = level.clip(new ClipContext(eye, end, ClipContext.Block.OUTLINE, ClipContext.Fluid.ANY, bee));
                    int rr, gg, bb, uv;
                    if (hit.getType() == HitResult.Type.MISS) {
                        // sky (darker below the horizon = far ground haze)
                        rr = elev > 0 ? 140 : 90; gg = elev > 0 ? 190 : 110; bb = elev > 0 ? 255 : 130; uv = elev > 0 ? 160 : 40;
                    } else {
                        BlockPos p = hit.getBlockPos();
                        BlockState s = level.getBlockState(p);
                        int col24 = s.getMapColor(level, p).col;
                        double fog = 1.0 - 0.6 * hit.getLocation().distanceTo(eye) / VIEW_DIST;
                        rr = (int) (((col24 >> 16) & 255) * fog); gg = (int) (((col24 >> 8) & 255) * fog); bb = (int) ((col24 & 255) * fog);
                        Long refill = emptyFlowers.get(p);
                        uv = isFlower(s) ? ((refill != null && refill > tick) ? 80 : 255) : 0;   // flowers glow in UV, like nectar guides
                    }
                    out[k++] = (byte) rr; out[k++] = (byte) gg; out[k++] = (byte) bb; out[k++] = (byte) uv;
                }
            }
        }
        return out;
    }

    static float wrap(float a) {
        a %= 360f;
        if (a > 180f) a -= 360f;
        if (a < -180f) a += 360f;
        return a;
    }

    // ------------------------------------------------------------------ brain link thread
    static void brainLink() {
        while (true) {
            try (Socket s = new Socket()) {
                s.connect(new InetSocketAddress("127.0.0.1", PORT), 1000);
                s.setTcpNoDelay(true);
                s.setSoTimeout(5000);
                OutputStream out = s.getOutputStream();
                BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), StandardCharsets.US_ASCII));
                connected = true;
                while (true) {
                    String obs = latestObs.getAndSet(null);
                    if (obs == null) { Thread.sleep(2); continue; }
                    out.write(obs.getBytes(StandardCharsets.US_ASCII));
                    out.flush();
                    String reply = in.readLine();
                    if (reply == null) break;
                    String[] p = reply.trim().split(" ");
                    if (p.length >= 4 && p[0].equals("M")) {
                        turn = Float.parseFloat(p[1]);
                        speed = Float.parseFloat(p[2]);
                        dash = p[3].equals("1");
                    }
                }
            } catch (Exception ignored) {
            }
            connected = false; turn = 0f; dash = false;
            try { Thread.sleep(2000); } catch (InterruptedException e) { return; }
        }
    }
}
