# production/network_view.py
# =============================================================================
# AdapSecMAS — NetworkView
# Pygame window 2: network dashboard.
#   - Agent graph with SNR-coloured links
#   - Per-agent queue bars
#   - Security level indicator per agent
#   - Active protocols overlay
#   - Attack indicators (ban list, CRL)
#
# SRP: rendering only — reads state dict from DeliveryEnv.
# =============================================================================

from __future__ import annotations

import math
import pygame

from core.constants  import N_AGENTS
from security.levels import SecurityLevel


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

BG          = (20,  22,  30)
TEXT        = (220, 220, 220)
TEXT_DIM    = (120, 120, 120)
LINK_GOOD   = ( 29, 158, 117)    # SNR well above threshold
LINK_WARN   = (239, 159,  39)    # SNR near threshold
LINK_BAD    = (226,  75,  74)    # SNR below threshold
AGENT_FILL  = ( 40,  44,  55)

LEVEL_COLS = {
    SecurityLevel.NORMAL  : ( 29, 158, 117),
    SecurityLevel.ELEVATED: (239, 159,  39),
    SecurityLevel.HIGH    : (226,  75,  74),
    SecurityLevel.CRITICAL: (123,  45, 139),
}

PANEL_BG    = ( 30,  33,  44)
BAR_BG      = ( 50,  53,  64)

W, H = 560, 700   # window size


class NetworkView:
    """
    Renders the network topology and security dashboard.

    Layout:
      Left (360px)  : agent graph — nodes + SNR-coloured links
      Right (200px) : per-agent panel — queue bar, level, action

    SRP: rendering only.
    """

    def __init__(self, title: str = "AdapSecMAS — Network"):
        pygame.init()
        self._screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption(title)
        self._clock  = pygame.time.Clock()
        self._font_s = pygame.font.SysFont("monospace", 11)
        self._font_m = pygame.font.SysFont("monospace", 13, bold=True)
        self._font_l = pygame.font.SysFont("monospace", 16, bold=True)

        # Pre-compute circular layout for 20 agents
        self._node_positions = self._circular_layout(
            n      = N_AGENTS,
            cx     = 180,
            cy     = 350,
            radius = 145,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def render(self, state: dict, fps: int = 60) -> bool:
        """
        Render one frame.
        Returns False if the window was closed.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        self._screen.fill(BG)
        self._draw_title(state)
        self._draw_links(state)
        self._draw_nodes(state)
        self._draw_panel(state)
        self._draw_attack_indicators(state)

        pygame.display.flip()
        self._clock.tick(fps)
        return True

    def close(self) -> None:
        pygame.quit()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_title(self, state: dict) -> None:
        info = state.get("info", {})
        t    = self._font_l.render("AdapSecMAS — Network", True, TEXT)
        self._screen.blit(t, (10, 8))

        reward_txt = self._font_s.render(
            f"reward={state.get('reward', 0.0):.2f}  "
            f"t={state.get('sim_time', 0.0):.1f}s",
            True, TEXT_DIM
        )
        self._screen.blit(reward_txt, (10, 32))

    def _draw_links(self, state: dict) -> None:
        """Draw agent-to-agent links coloured by SNR."""
        snr_map   = state.get("network_state", {}).get("snr", {})
        banned    = state.get("network_state", {}).get("banned", {})
        threshold = 4.0

        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                if i in banned or j in banned:
                    continue

                snr = snr_map.get((i, j), snr_map.get((j, i), 0.0))
                if snr < 0.5:
                    continue   # no link

                col = (
                    LINK_GOOD if snr >= threshold * 1.5 else
                    LINK_WARN if snr >= threshold        else
                    LINK_BAD
                )
                alpha = min(200, int(80 + snr * 8))
                xi, yi = self._node_positions[i]
                xj, yj = self._node_positions[j]

                link_surf = pygame.Surface((W, H), pygame.SRCALPHA)
                pygame.draw.line(link_surf, (*col, alpha), (xi, yi), (xj, yj), 1)
                self._screen.blit(link_surf, (0, 0))

    def _draw_nodes(self, state: dict) -> None:
        """Draw agent nodes with security level colour ring."""
        obs       = state.get("obs", {})
        banned    = state.get("network_state", {}).get("banned", {})
        crl       = state.get("network_state", {}).get("crl", set())

        for i in range(N_AGENTS):
            x, y = self._node_positions[i]

            # Security level from obs
            level_norm = float(obs.get(i, [0] * 7)[6]) if obs else 0.0
            level_idx  = min(3, round(level_norm * 3))
            level      = SecurityLevel(level_idx)
            ring_col   = LEVEL_COLS[level]

            # Banned or revoked agents shown differently
            if i in banned:
                pygame.draw.circle(self._screen, (226, 75, 74), (x, y), 13, 3)
                pygame.draw.line(self._screen, (226, 75, 74), (x-9, y-9), (x+9, y+9), 2)
                pygame.draw.line(self._screen, (226, 75, 74), (x+9, y-9), (x-9, y+9), 2)
                continue

            # Normal node
            pygame.draw.circle(self._screen, ring_col, (x, y), 13, 2)
            pygame.draw.circle(self._screen, AGENT_FILL, (x, y), 11)

            label = self._font_s.render(str(i), True, TEXT)
            lw    = label.get_width()
            self._screen.blit(label, (x - lw // 2, y - 6))

    def _draw_panel(self, state: dict) -> None:
        """Right panel: per-agent queue bar and level indicator."""
        obs       = state.get("obs", {})
        net_state = state.get("network_state", {})
        banned    = net_state.get("banned", {})

        panel_x = 370
        pygame.draw.rect(self._screen, PANEL_BG, (panel_x, 0, W - panel_x, H))

        title = self._font_m.render("Agents", True, TEXT)
        self._screen.blit(title, (panel_x + 10, 10))

        row_h  = 30
        start_y = 40

        for i in range(N_AGENTS):
            y        = start_y + i * row_h
            agent_obs= obs.get(i, [0.0] * 12) if obs else [0.0] * 12

            # Level
            level_norm = float(agent_obs[6])
            level_idx  = min(3, round(level_norm * 3))
            level      = SecurityLevel(level_idx)
            col        = LEVEL_COLS[level]

            # Level dot
            pygame.draw.circle(self._screen, col, (panel_x + 14, y + 10), 6)

            # Agent ID
            id_txt = self._font_s.render(f"A{i:02d}", True, TEXT if i not in banned else (226, 75, 74))
            self._screen.blit(id_txt, (panel_x + 24, y + 4))

            # SNR bar
            snr_norm = float(agent_obs[0])
            self._draw_mini_bar(
                x=panel_x + 58, y=y + 8,
                width=60, height=8,
                value=snr_norm,
                col=LINK_GOOD if snr_norm > 0.3 else LINK_BAD,
            )

            # Flood indicator
            flood = float(agent_obs[3])
            if flood > 0.3:
                f_txt = self._font_s.render("FLD", True, (239, 159, 39))
                self._screen.blit(f_txt, (panel_x + 124, y + 4))

            # Spoof flag
            if float(agent_obs[4]) > 0.5:
                s_txt = self._font_s.render("SPF", True, (123, 45, 139))
                self._screen.blit(s_txt, (panel_x + 155, y + 4))

    def _draw_attack_indicators(self, state: dict) -> None:
        """Bottom area: active protocols, ban list, CRL."""
        net_state = state.get("network_state", {})
        banned    = net_state.get("banned", {})
        crl       = net_state.get("crl", set())
        channel   = net_state.get("channel", {})

        y = H - 90
        pygame.draw.rect(self._screen, PANEL_BG, (0, y, 360, 90))

        title = self._font_m.render("Security Events", True, TEXT)
        self._screen.blit(title, (10, y + 6))

        if banned:
            ban_txt = self._font_s.render(
                f"BANNED: {list(banned.keys())}", True, (226, 75, 74)
            )
            self._screen.blit(ban_txt, (10, y + 26))

        if crl:
            crl_txt = self._font_s.render(
                f"CRL: {list(crl)}", True, (123, 45, 139)
            )
            self._screen.blit(crl_txt, (10, y + 44))

        # Channel hop indicator
        channels = set(channel.values())
        if len(channels) > 1:
            ch_txt = self._font_s.render(
                f"FREQ-HOP active  ch={list(channels)}", True, (29, 158, 117)
            )
            self._screen.blit(ch_txt, (10, y + 62))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_mini_bar(
        self,
        x: int, y: int,
        width: int, height: int,
        value: float,
        col: tuple,
    ) -> None:
        pygame.draw.rect(self._screen, BAR_BG, (x, y, width, height))
        pygame.draw.rect(
            self._screen, col,
            (x, y, int(width * max(0.0, min(1.0, value))), height)
        )

    @staticmethod
    def _circular_layout(
        n: int, cx: int, cy: int, radius: int
    ) -> list[tuple[int, int]]:
        """Arrange n nodes in a circle."""
        positions = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x     = cx + int(radius * math.cos(angle))
            y     = cy + int(radius * math.sin(angle))
            positions.append((x, y))
        return positions