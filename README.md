# Black Hole Ray Tracer 🌌
An interactive real-time visualization of photon trajectories around a Schwarzschild black hole. Experience Einstein's theory of general 
relativity through accurate numerical simulations of light bending in curved spacetime.

✨ Features
-Real-time Physics: Accurate numerical integration of null geodesics in Schwarzschild spacetime
-Interactive Photon Launching: Click anywhere to launch photons with different initial trajectories
-Multiple Motion Types: Horizontal, vertical, tangential, and radial photon paths
-GPU Acceleration: Optional CuPy support for massive performance boosts
-Visual Effects: Color-coded photon trails with fading transparency
-Educational HUD: Real-time information display and physics parameters

Prerequisites
-Python 3.8 or higher
-Pygame
-NumPy

🎮 Controls
Action                  	Control
Launch Photon          	Left Click
Horizontal Motion	      Left Click (default)
Vertical Motion	        Shift + Left Click
Tangential Motion      	Ctrl + Left Click
Radial Outward Motion	  Alt + Left Click
Random Photon           Spacebar
Camera Pan	            Middle Mouse Drag
Zoom	                  Mouse Wheel
Reset Simulation	     R Key
Toggle Auto-spawn	     A Key
Quit	                 ESC Key

🔬 Physics Behind the Simulation
Schwarzschild Metric
The simulation solves the geodesic equation for light in Schwarzschild spacetime:

ds² = -(1 - 2GM/rc²)c²dt² + (1 - 2GM/rc²)⁻¹dr² + r²(dθ² + sin²θ dφ²)

Key Features
-Event Horizon: At r = 2GM/c² - boundary of no return
-Photon Sphere: At r = 3GM/c² - unstable light orbits
-Gravitational Lensing: Light bending around massive objects
-Conserved Quantities: Energy (E) and angular momentum (L) preservation

