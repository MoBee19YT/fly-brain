// Fly Brain - lets a fruit-fly connectome play Geometry Dash.
//
// Every physics tick this mod sends the player's state to the local fly brain server
// (fly_server.py, 127.0.0.1:5577) and presses / releases the button according to the reply.
// When a level starts it also sends the level's hazards, solids and portals once.
// If the server is not running, the game plays normally.

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>

#include <string>
#include <sstream>
#include <unordered_set>
#include <chrono>

#ifdef _WIN32
    using sock_t = SOCKET;
    static constexpr sock_t BAD_SOCK = INVALID_SOCKET;
    static void closeSock(sock_t s) { closesocket(s); }
#else
    #include <sys/socket.h>
    #include <sys/time.h>
    #include <sys/select.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    using sock_t = int;
    static constexpr sock_t BAD_SOCK = -1;
    static void closeSock(sock_t s) { close(s); }
#endif

using namespace geode::prelude;

// ------------------------------------------------------------------ connection
class BrainLink {
    sock_t m_sock = BAD_SOCK;
    std::string m_buf;
public:
    bool connected() const { return m_sock != BAD_SOCK; }

    bool connect(int port) {
        disconnect();
#ifdef _WIN32
        static bool wsa = false;
        if (!wsa) { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); wsa = true; }
#endif
        sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (s == BAD_SOCK) return false;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            closeSock(s);
            return false;
        }
        int one = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
#ifdef __APPLE__
        // without this, macOS kills the whole game (SIGPIPE) if the brain server closes mid-level
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
        m_sock = s;
        setTimeout(1500);
        m_buf.clear();
        return true;
    }

    void setTimeout(int ms) {
        if (!connected()) return;
#ifdef _WIN32
        DWORD tv = static_cast<DWORD>(ms);
        setsockopt(m_sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
        timeval tv{ms / 1000, (ms % 1000) * 1000};
        setsockopt(m_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
    }

    std::string lastError;

    void disconnect() {
        if (m_sock != BAD_SOCK) closeSock(m_sock);
        m_sock = BAD_SOCK;
    }

    bool send(std::string const& msg) {
        if (!connected()) return false;
        size_t off = 0;
        while (off < msg.size()) {
            int n = ::send(m_sock, msg.data() + off, static_cast<int>(msg.size() - off), 0);
            if (n <= 0) { log::warn("Fly brain: send failed - disconnecting"); disconnect(); return false; }
            off += static_cast<size_t>(n);
        }
        return true;
    }

    // Returns true if a full line is available right now (never waits). Used in async mode.
    bool pollLine(std::string& out) {
        if (!connected()) return false;
        while (true) {
            auto pos = m_buf.find('\n');
            if (pos != std::string::npos) {
                out = m_buf.substr(0, pos);
                m_buf.erase(0, pos + 1);
                return true;
            }
            fd_set rd; FD_ZERO(&rd); FD_SET(m_sock, &rd);
            timeval tv{0, 0};
            int ready = select(static_cast<int>(m_sock) + 1, &rd, nullptr, nullptr, &tv);
            if (ready <= 0) return false;
            char tmp[256];
            int n = ::recv(m_sock, tmp, sizeof(tmp), 0);
            if (n <= 0) { lastError = "server closed the connection"; disconnect(); return false; }
            m_buf.append(tmp, static_cast<size_t>(n));
        }
    }

    // Reads one line. Returns false on timeout / error.
    bool readLine(std::string& out) {
        if (!connected()) return false;
        while (true) {
            auto pos = m_buf.find('\n');
            if (pos != std::string::npos) {
                out = m_buf.substr(0, pos);
                m_buf.erase(0, pos + 1);
                return true;
            }
            char tmp[256];
            int n = ::recv(m_sock, tmp, sizeof(tmp), 0);
            if (n <= 0) {
                lastError = n == 0 ? "server closed the connection" : "no answer from the brain in time";
                log::warn("Fly brain: {} - disconnecting", lastError);
                disconnect();
                return false;
            }
            m_buf.append(tmp, static_cast<size_t>(n));
        }
    }
};

static BrainLink g_link;
static bool g_holding = false;
static double g_time = 0.0;
static double g_lastAsk = -1.0;   // game time of the last question to the brain
static bool g_async = false;      // brain too slow to wait for (real fly mode): game never waits
static bool g_inFlight = false;   // async: a question is waiting for its answer
static double g_askedAt = 0.0;    // async: real time the pending question was sent

static bool enabled() { return Mod::get()->getSettingValue<bool>("enabled"); }
static double trainingSpeed() { return Mod::get()->getSettingValue<double>("speed"); }

static void applySpeed(bool on) {
    float s = (on && enabled() && g_link.connected()) ? static_cast<float>(trainingSpeed()) : 1.0f;
    CCDirector::get()->getScheduler()->setTimeScale(s);
}
static int port() { return static_cast<int>(Mod::get()->getSettingValue<int64_t>("port")); }

static int playerMode(PlayerObject* p) {
    if (p->m_isShip) return 1;
    if (p->m_isDart) return 2;   // wave
    if (p->m_isBall) return 3;
    if (p->m_isBird) return 4;   // ufo
    if (p->m_isRobot) return 5;
    if (p->m_isSpider) return 6;
    if (p->m_isSwing) return 7;
    return 0;                    // cube
}

// Portal / speed object IDs the brain server cares about (it also gets every solid and hazard).
static std::unordered_set<int> const SPECIAL_IDS = {
    12, 13, 47, 111, 660, 745, 1331, 1933,     // cube, ship, ball, ufo, wave, robot, spider, swing portals
    99, 101,                                    // normal / mini size portals
    10, 11, 2926,                               // gravity portals
    200, 201, 202, 203, 1334,                   // speed portals 0.5x, 1x, 2x, 3x, 4x
};

static void sendLevel(GJBaseGameLayer* layer) {
    std::ostringstream ss;
    int count = 0;
    std::ostringstream body;
    body.precision(2);
    body << std::fixed;
    for (auto obj : CCArrayExt<GameObject*>(layer->m_objects)) {
        if (!obj) continue;
        auto type = static_cast<int>(obj->m_objectType);
        bool keep = type == static_cast<int>(GameObjectType::Solid)
                 || type == static_cast<int>(GameObjectType::Hazard)
                 || type == static_cast<int>(GameObjectType::AnimatedHazard)
                 || type == static_cast<int>(GameObjectType::Slope)
                 || SPECIAL_IDS.contains(obj->m_objectID);
        if (!keep) continue;
        auto const& r = obj->getObjectRect();
        body << "O " << obj->m_objectID << ' ' << type << ' '
             << r.origin.x << ' ' << r.origin.y << ' ' << r.size.width << ' ' << r.size.height << '\n';
        count++;
    }
    ss << "L " << count << '\n' << body.str() << "E\n";
    g_link.send(ss.str());
    log::info("Fly brain: sent level with {} objects", count);
}

static void tryConnect(PlayLayer* layer) {
    if (!enabled()) return;
    if (g_link.connect(port())) {
        g_link.send("H 3\n");
        if (auto lvl = layer->m_level) {
            std::string name = lvl->m_levelName;
            g_link.send(fmt::format("I {} {}\n", lvl->m_levelID.value(), name));
        }
        sendLevel(layer);
        // wait (up to 10 s) until the brain has the level and is ready
        g_link.setTimeout(10000);
        std::string ack;
        if (!g_link.readLine(ack) || ack.empty() || ack[0] != 'K') {
            log::warn("Fly brain: server did not confirm the level ('{}')", ack);
            g_link.disconnect();
            return;
        }
        g_link.setTimeout(1500);
        g_async = ack.find("async") != std::string::npos;
        g_inFlight = false;
        log::info("Fly brain connected on port {} ({} mode)", port(), g_async ? "async" : "sync");
    } else {
        log::info("Fly brain server not running on port {} - playing normally", port());
    }
}

static void releaseButton(GJBaseGameLayer* layer) {
    if (g_holding) {
        layer->handleButton(false, 1, true);
        g_holding = false;
    }
}

// ------------------------------------------------------------------ hooks
class $modify(FlyPlayLayer, PlayLayer) {
#if GEODE_COMP_GD_VERSION >= 22090
    // GD 2.209+ (Geode v6): init lost its useReplay argument
    bool init(GJGameLevel* level, bool dontCreateObjects) {
        if (!PlayLayer::init(level, dontCreateObjects)) return false;
#else
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
#endif
        g_holding = false;
        g_time = 0.0;
        g_lastAsk = -1.0;
        g_inFlight = false;
        tryConnect(this);
        applySpeed(true);
        return true;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        // GD "kills" the player with a hidden anticheat spike at level start - ignore that one
        bool real = player == m_player1 && object != m_anticheatSpike && player && !player->m_isDead;
        PlayLayer::destroyPlayer(player, object);
        if (real && player->m_isDead && g_link.connected()) {
            g_link.send(fmt::format("D {:.2f}\n", player->getPositionX()));
        }
    }

    void levelComplete() {
        if (g_link.connected()) g_link.send("W\n");
        PlayLayer::levelComplete();
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        g_holding = false;
        g_time = 0.0;
        g_lastAsk = -1.0;
        g_inFlight = false;
        if (!g_link.connected()) { tryConnect(this); applySpeed(true); }
        g_link.send("R\n");
    }

    void onQuit() {
        releaseButton(this);
        CCDirector::get()->getScheduler()->setTimeScale(1.0f);
        g_link.send("Q\n");
        g_link.disconnect();
        PlayLayer::onQuit();
    }
};

class $modify(FlyGameLayer, GJBaseGameLayer) {
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        auto pl = PlayLayer::get();
        if (pl && static_cast<GJBaseGameLayer*>(pl) == this && g_link.connected() && enabled()) {
            auto p = m_player1;
            if (p && !p->m_isDead) {
                g_time += dt;
            }
            if (g_async) {
                // never wait: pick up an answer if one has arrived, and ask again when the last one is done
                std::string reply;
                while (g_link.pollLine(reply)) {
                    g_inFlight = false;
                    bool hold = !reply.empty() && reply[0] == '1';
                    if (hold != g_holding) { this->handleButton(hold, 1, true); g_holding = hold; }
                }
                double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                if (g_inFlight && now - g_askedAt > 5.0) { log::warn("Fly brain: no answer for 5 s"); g_link.disconnect(); }
                if (p && !p->m_isDead && g_link.connected() && !g_inFlight) {
                    auto pos = p->getPosition();
                    g_link.send(fmt::format("S {:.4f} {:.4f} {:.4f} {:.4f} {} {} {} {} {:.4f}\n", g_time, pos.x, pos.y,
                        p->m_yVelocity, playerMode(p), p->m_vehicleSize < 0.9f ? 1 : 0, p->m_isOnGround ? 1 : 0,
                        p->m_isUpsideDown ? 1 : 0, p->m_playerSpeed));
                    g_inFlight = true; g_askedAt = now;
                }
            }
            // the brain decides 60 times per game-second; in between, keep the last decision
            else if (p && !p->m_isDead && (g_lastAsk < 0 || g_time - g_lastAsk >= 1.0 / 60.0 - 1e-6)) {
                g_lastAsk = g_time;
                auto pos = p->getPosition();
                std::ostringstream ss;
                ss.precision(4);
                ss << std::fixed << "S " << g_time << ' ' << pos.x << ' ' << pos.y << ' '
                   << p->m_yVelocity << ' ' << playerMode(p) << ' '
                   << (p->m_vehicleSize < 0.9f ? 1 : 0) << ' ' << (p->m_isOnGround ? 1 : 0) << ' '
                   << (p->m_isUpsideDown ? 1 : 0) << ' ' << p->m_playerSpeed << '\n';
                std::string reply;
                if (g_link.send(ss.str()) && g_link.readLine(reply) && !reply.empty()) {
                    bool hold = reply[0] == '1';
                    if (hold != g_holding) {
                        this->handleButton(hold, 1, true);
                        g_holding = hold;
                    }
                }
            }
        }
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
    }
};
