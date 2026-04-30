# production/map_view.py
# =============================================================================
# AdapSecMAS — MapView
# Pygame window 1: city map with jammer heatmap, trucks, drones, packages.
# SRP: only responsible for rendering the spatial view.
#      No simulation logic, no network logic.
# =============================================================================

from __future__ import annotations

import math
import pygame

from core.constants     import ARENA_WIDTH, ARENA_HEIGHT
from security.levels    import SecurityLevel


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BG          = (245, 245, 240)
ROAD        = (200, 200, 195)
BUILDING    = (160, 155, 150)
DEPOT       = ( 83,  74, 183)   # purple
TRUCK_COL   = ( 29, 158, 117)   # green
DRONE_COL   = ( 29, 158, 117)
PKG_PENDING = (239, 159,  39)   # orange
PKG_TRANSIT = ( 29, 158, 117)
PKG_DONE    = (130, 130, 130)
PKG_FAILED  = (226,  75,  74)   # red
TEXT_COL    = ( 40,  40,  40)
HUD_BG      = (255, 255, 255, 180)

# Security level colours
LEVEL_COLS = {
    SecurityLevel.NORMAL  : ( 29, 158, 117),
    SecurityLevel.ELEVATED: (239, 159,  39),
    SecurityLevel.HIGH    : (226,  75,  74),
    SecurityLevel.CRITICAL: (123,  45, 139),
}


class MapView:
    """
    Renders the city delivery map in a pygame window.

    Layers (bottom to top):
      1. Background + roads
      2. Jammer heatmap (transparent overlay)
      3. Package destinations
      4. Delivery agents (trucks + drones)
      5. HUD (stats bar)

    SRP: rendering only — reads state dict from DeliveryEnv.
    """

    WINDOW_W = int(ARENA_WIDTH)
    WINDOW_H = int(ARENA_HEIGHT) + 60   # +60 for HUD bar

    def __init__(self, title: str = "AdapSecMAS — City Map"):
        pygame.init()
        self._screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption(title)
        self._clock  = pygame.time.Clock()
        self._font_s = pygame.font.SysFont("monospace", 12)
        self._font_m = pygame.font.SysFont("monospace", 14, bold=True)

        # Pre-build heatmap surface (reused each frame)
        self._heatmap = pygame.Surface(
            (self.WINDOW_W, int(ARENA_HEIGHT)), pygame.SRCALPHA
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, state: dict, fps: int = 60) -> bool:
        """
        Render one frame.
        Returns False if the window close button was pressed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self._screen.fill(BG)
        self._draw_roads()
        self._draw_jammer_heatmap(state)
        self._draw_packages(state)
        self._draw_depot(state)
        self._draw_agents(state)
        self._draw_hud(state)

        pygame.display.flip()
        self._clock.tick(fps)
        return True

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_roads(self) -> None:
        """Simple grid road pattern."""
        for x in range(0, self.WINDOW_W, 140):
            pygame.draw.line(self._screen, ROAD, (x, 0), (x, int(ARENA_HEIGHT)), 20)
        for y in range(0, int(ARENA_HEIGHT), 120):
            pygame.draw.line(self._screen, ROAD, (0, y), (self.WINDOW_W, y), 20)

    def _draw_jammer_heatmap(self, state: dict) -> None:
        """
        Draw the jammer noise field as a radial gradient overlay.
        Transparency proportional to noise intensity.
        """
        jpos   = state.get("jammer_pos",    (0, 0))
        jrad   = state.get("jammer_radius", 430)

        self._heatmap.fill((0, 0, 0, 0))

        # Radial gradient — concentric circles with decreasing alpha
        steps = 12
        for k in range(steps, 0, -1):
            r     = int(jrad * k / steps)
            alpha = int(120 * (1 - k / steps))
            surf  = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (226, 75, 74, alpha), (r, r), r)
            self._heatmap.blit(
                surf,
                (int(jpos[0]) - r, int(jpos[1]) - r),
            )

        self._screen.blit(self._heatmap, (0, 0))

        # Jammer centre dot
        pygame.draw.circle(
            self._screen, (180, 30, 30),
            (int(jpos[0]), int(jpos[1])), 8
        )
        label = self._font_s.render("JAM", True, (180, 30, 30))
        self._screen.blit(label, (int(jpos[0]) + 10, int(jpos[1]) - 8))

    def _draw_packages(self, state: dict) -> None:
        """Draw package destinations as small coloured diamonds."""
        for pkg in state.get("packages", []):
            col = {
                "PENDING"   : PKG_PENDING,
                "ASSIGNED"  : PKG_PENDING,
                "IN_TRANSIT": PKG_TRANSIT,
                "DELIVERED" : PKG_DONE,
                "FAILED"    : PKG_FAILED,
            }.get(pkg.status.name, PKG_PENDING)

            x, y = int(pkg.destination[0]), int(pkg.destination[1])
            pts  = [(x, y - 7), (x + 5, y), (x, y + 7), (x - 5, y)]
            pygame.draw.polygon(self._screen, col, pts)

    def _draw_depot(self, state: dict) -> None:
        """Draw the central depot."""
        dx, dy = state.get("depot_pos", (80, 80))
        pygame.draw.rect(
            self._screen, DEPOT,
            (int(dx) - 15, int(dy) - 15, 30, 30), border_radius=4
        )
        label = self._font_s.render("DEPOT", True, DEPOT)
        self._screen.blit(label, (int(dx) - 18, int(dy) + 18))

    def _draw_agents(self, state: dict) -> None:
        """Draw trucks and drones with security level colour ring."""
        net_state = state.get("network_state", {})
        obs       = state.get("obs", {})

        for agent in state.get("agents", []):
            x, y = int(agent.pos[0]), int(agent.pos[1])

            # Security level ring colour from SLM level_norm in obs
            level_norm = float(obs.get(agent.agent_id, [0] * 7)[6]) if obs else 0.0
            level_idx  = min(3, round(level_norm * 3))
            level      = SecurityLevel(level_idx)
            ring_col   = LEVEL_COLS[level]

            if agent.agent_type.name == "TRUCK":
                # Truck: filled rectangle
                pygame.draw.rect(self._screen, ring_col, (x - 12, y - 7, 24, 14), 2)
                pygame.draw.rect(self._screen, TRUCK_COL, (x - 10, y - 5, 20, 10))
            else:
                # Drone: circle
                pygame.draw.circle(self._screen, ring_col, (x, y), 11, 2)
                pygame.draw.circle(self._screen, DRONE_COL, (x, y), 8)

            # Agent ID label
            label = self._font_s.render(str(agent.agent_id), True, TEXT_COL)
            self._screen.blit(label, (x + 10, y - 8))

            # Battery bar
            bw = 20
            bh = 3
            pygame.draw.rect(self._screen, (200, 200, 200), (x - bw // 2, y + 12, bw, bh))
            pygame.draw.rect(
                self._screen,
                (50, 200, 50) if agent.battery > 0.3 else (220, 80, 80),
                (x - bw // 2, y + 12, int(bw * agent.battery), bh),
            )

    def _draw_hud(self, state: dict) -> None:
        """Draw stats bar at the bottom of the window."""
        hud_y = int(ARENA_HEIGHT)
        pygame.draw.rect(
            self._screen, (230, 230, 230),
            (0, hud_y, self.WINDOW_W, 60)
        )

        info     = state.get("info", {})
        packages = state.get("packages", [])
        delivered= sum(1 for p in packages if p.status.name == "DELIVERED")
        total    = len(packages)
        reward   = state.get("reward", 0.0)
        sim_time = state.get("sim_time", 0.0)

        texts = [
            f"t={sim_time:.1f}s",
            f"packages={delivered}/{total}",
            f"reward={reward:.1f}",
            f"jam_loss={info.get('n_msgs_lost_to_jam', 0)}",
            f"overflow={info.get('n_queue_overflows', 0)}",
            f"spoof={info.get('n_spoof_accepted', 0)}",
        ]
        x = 10
        for text in texts:
            surf = self._font_m.render(text, True, TEXT_COL)
            self._screen.blit(surf, (x, hud_y + 20))
            x += surf.get_width() + 20