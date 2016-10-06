CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

## Summary
For this project, I implemented part of a pathtracer, a program for rendering an image of a 3D scene based on a specification with locations of shapes in space. Unlike a physical camera, which perceives objects by receiving light that has bounced off of them from a light source, a pathtracer follows the path of light in reverse: on each iteration, the program projects rays from the camera toward each pixel on the screen. With each time-step<sup id="a1">[1](#f1)</sup>, the ray bounces off a surface and samples colors from it. The ray terminates if it strikes a light source, runs out of bounces, or strikes empty space.

## Basic Features
### Scatter mechanism
Once a ray strikes an object (besides a light source), it bounces and a new path (origin and direction) must be calculated for it. My scatter function handles multiple cases:
- When a material is fully reflective (ideal specular). This is the simplest of the cases: the ray is simply reflected such that the angle of incidence equals the angle of reflection.
- When a material is fully diffuse, the ray bounces off at a completely random angle within the hemisphere defined by the surface normal. For example, if a ray bounces off an ordinary wall surface, it's new path is in a completely random direction, excluding paths that actually penetrate the wall.
- When a material is diffractive, light penetrates the surface but bends based on the refraction index of the material <sup id="a2">[2](#f2)</sup>.
- When a material is both refractive and reflective, the ray chooses randomly between refraction and reflection, using a distribution defined by the properties of the material (its ratio of "hasRefractive" to "hasReflective").

### Shading mechanism
A ray begins as white and as it strikes a material, multiplies its current color by the material's color. If the ray strikes a light source, the accumulated color value is multiplied by the brightness of the light. If it strikes empty space or runs out of bounces, the color is set to black. This effect accounts for shadows, because surfaces that do not have a direct path to light have a lower chance of reflecting a ray into the light.

## Optimizations
### Ray compaction
Every time-step, rays may terminate by striking empty space or a light. A naive approach to handling these rays would be to set a flag indicating that they are no longer active and then check this flag at the start of the shading kernel (to prevent further coloration). The problem with this approach is that the threads assigned to these dead rays would be _wasted_. Instead, we perform stream compaction on the rays at the end of every time step to eliminate

<b id="f1">1</b> The distinction between iterations and time-steps may be a little confusing. Within a time-step, a light ray bounces (at most) once -- it moves from one surface to another or strikes empty space. In contrast, an iteration is only complete once all rays have terminated. This generally involves multiple time-steps and bounces. [↩](#a1)

<b id="f2">2</b> Technically the angle is defined by the _ratio_ of the refraction indices of the substances involved, e.g. air to water if the ray is entering water from the air. [↩](#a2)
