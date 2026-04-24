# fly_brain/hardware/__init__.py
"""
Hardware I/O layer for fly_brain.

Provides GPIO-backed sensor drivers and ESC controllers that fall back
to simulation mode automatically when running off-hardware (no RPi.GPIO).

Submodules
----------
ultrasonic_driver   UltrasonicArray  — HC-SR04 sensors via GPIO
esc_output          ESCController    — Motor ESC via GPIO PWM or PCA9685
flybrainIO          FlyBrainIO       — 100 Hz main I/O loop
"""
