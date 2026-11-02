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

#ifdef _WIN32
    using sock_t = SOCKET;
    static constexpr sock_t BAD_SOCK = INVALID_SOCKET;
    static void closeSock(sock_t s) { closesocket(s); }
#else
    #include <sys/socket.h>
    #include <sys/time.h>
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
#ifdef _WIN32
        DWORD tv = 200;
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
        timeval tv{0, 200000};
        setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
        m_sock = s;
        m_buf.clear();
        return true;
    }

    void disconnect() {
        if (m_sock != BAD_SOCK) closeSock(m_sock);
        m_sock = BAD_SOCK;
    }

    bool send(std::string const& msg) {
        if (!connected()) return false;
        size_t off = 0;
        while (off < msg.size()) {
            int n = ::send(m_sock, msg.data() + off, static_cast<int>(msg.size() - off), 0);
            if (n <= 0) { disconnect(); return false; }
            off += static_cast<size_t>(n);
        }
        return true;
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
            if (n <= 0) { disconnect(); return false; }
            m_buf.append(tmp, static_cast<size_t>(n));
        }
    }
};

static BrainLink g_link;
static bool g_holding = false;
static double g_time = 0.0;

static bool enabled() { return Mod::get()->getSettingValue<bool>("enabled"); }
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

static void tryConnect(GJBaseGameLayer* layer) {
    if (!enabled()) return;
    if (g_link.connect(port())) {
        g_link.send("H 1\n");
        sendLevel(layer);
        log::info("Fly brain connected on port {}", port());
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
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;
        g_holding = false;
        g_time = 0.0;
        tryConnect(this);
        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        g_holding = false;
        g_time = 0.0;
        if (!g_link.connected()) tryConnect(this);
        g_link.send("R\n");
    }

    void onQuit() {
        releaseButton(this);
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
