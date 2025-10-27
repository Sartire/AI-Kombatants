
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99 &
python3 -m retro.scripts.playback_movie MortalKombatII-Genesis-Level1.JaxVsBaraka-000000.bk2