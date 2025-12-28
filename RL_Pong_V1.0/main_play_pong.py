# main_play_pong.py
"""
Main Entry Point - Play Pong GUI
Simple wrapper around the playable GUI.

Usage:
    python main_play_pong.py

Controls:
    ↑ / ↓ ARROW : Move paddle
    R           : Restart game
    D           : Toggle debug info
    ESC         : Quit

Author: [Your Name]
Date: 2025
"""
import sys
import pygame
from ui_playable_pong_gui import PlayablePongGUI


def main():
    """Entry point for playable Pong GUI"""
    try:
        game = PlayablePongGUI()
        game.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Game interrupted by user")
        pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\n⚠ Error running game: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)


if __name__ == "__main__":
    main()