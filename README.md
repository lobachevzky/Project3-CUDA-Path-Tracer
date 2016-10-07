CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

## Summary
For this project, I implemented part of a pathtracer, a program for rendering an image of a scene, given the locations and attributes of shapes in 3D space. Unlike a physical camera, which perceives objects by capturing light that has bounced off of them from a light source, a pathtracer follows the path of light in reverse: on each iteration, the program projects rays from the camera toward each pixel on the screen. With each time-step<sup id="1">[1](#f1)</sup>, the ray bounces off a surface and samples colors from it. The ray terminates if it strikes a light source, runs out of bounces, or strikes empty space.

## Basic Features
### Scatter mechanism
Once a ray strikes an object (besides a light source), it bounces and a new path (origin and direction) must be calculated for it. My scatter function handles multiple cases:
- When a material is fully reflective (ideal specular). This is the simplest of the cases: the ray is simply reflected such that the angle of incidence equals the angle of reflection.
- When a material is fully diffuse, the ray bounces off at a completely random angle within the hemisphere defined by the surface normal. For example, if a ray bounces off an ordinary wall surface, it's new path is in a completely random direction, excluding paths that actually penetrate the wall.
- When a material is diffractive, light penetrates the surface but bends based on the refraction index of the material <sup id="2">[2](#f2)</sup>.
- When a material is both refractive and reflective, the ray chooses randomly between refraction and reflection, using a distribution defined by the properties of the material (its ratio of "hasRefractive" to "hasReflective").

### Shading mechanism
A ray begins as white and as it strikes a material, multiplies its current color by the material's color. If the ray strikes a light source, the accumulated color value is multiplied by the brightness of the light. If it strikes empty space or runs out of bounces, the color is set to black. This effect accounts for shadows, because surfaces that do not have a direct path to light have a lower chance of reflecting a ray into the light.

## Optimizations
### Ray compaction
Every time-step, rays may terminate by striking empty space or a light. A naive approach to handling these rays would be to set a flag indicating that they are no longer active and then check this flag at the start of the shading kernel (to prevent further coloration). The problem with this approach is that the threads assigned to these dead rays would be _wasted_. Instead, we perform stream compaction on the rays at the end of every time step to eliminate dead rays.

A pitfall of this optimization (one which cost me many hours of debugging) is that stream-compaction mutates the compacted array. Consequently, dead rays must be saved somehow so that their colors can be rendered at the end of the iterations.

One naive approach is to make a second array of pointers to the array of rays. Then we perform all operations, including stream compaction on the array of pointers instead of the array of rays. That way when we perform stream compaction, we only eliminate the pointers, not the rays themselves. Finally, once all pointers have been eliminated, we use the original array to render the image. This approach is depicted in this graphic:

**TODO**

A more performant approach is to maintain a separate array of color values in addition to the array of rays. Whenever we terminate a ray, we first store its color in the color array. Finally we use the color array to render the final image. This approach is depicted here:

**TODO**


### Storing materials in contiguous memory
When a ray strikes a surface, we must access that surface's material from global memory. By sorting rays by material type, we can increase the chances that a material has already been cached by a previously processed ray. In order to achieve this, we used `thrust::sort_by_key` to sort the rays by the materials associated with their corresponding surface intersections. Unfortunately, we found that this did not considerably improve performance:

**TODO**

### Caching the first bounce
In a typical pathtracer, all rays follow the same path, from the camera to their assigned pixel, on the first bounce. Consequently it is unnecessary to recalculate this first bounce every time. If the `cache1stBounce` is set to 1, then the program saves caches the first segment in `dev_1stpath` and the first intersection in `dev_1stIntersect`. This noticeably speeds up the program as indicated by:

**TODO**

## Extra Features
### Refraction
As mentioned in the section on the scatter mechanism, the program implements refraction in addition to reflection and diffuse scattering. The program only handles cases where light enters a refractive material from air or enters air from a refractive material. When the ray enters the refractive material, a toggle in the ray struct is set to `1`. If the ray strikes a refractive material from inside an object (as indicated by the toggle), the ratio of the indices of refraction is inverted -- this causes the light to bend back toward it's original direction as depicted in this image:

**TODO**

Refraction is not a performance-intensive feature. There is nothing GPU-specific about this feature. One way, however, that ray-scattering might generally be optimized is by storing `ShadeableIntersection` structs, which store information about the point where a ray intersects a surface, and `PathSegment` structs, which store information about the segment of a ray associated with a single bounce, in shared memory each time-step. This way,  subsequent accesses of these structs do not require calls to global memory. This is not difficult to implement since for the duration of a time-step, the arrays containing these structs are not reshuffled at all.

### Depth of field
Because a lens can precisely focus at only one distance at a time, objects at different distances may appear out of focus. In order to implement this feature, we jittered the camera by applying a random, small offset to its position and then recalculating the direction of the ray from its new origin to its assigned pixel (not recalculating the direction just causes the entire image to become blurry). Here is a comparison of the image, with and without depth of field added:

**TODO**

I was also curious whether caching the first bounce would have negative effects on the depth of field effect. Without caching, the cameras is set to a new, random starting position on each iteration, whereas with caching, the camera always starts from the same random offset. Surprisingly, as the following comparison demonstrates, caching had no noticeable impact on the depth of blur effect

**TODO**

[no caching] [caching]

One nice feature about depth-of-field is that it has absolutely no impact on performance, although images employing depth-of-field benefit significantly from antialiasing, which does come at a significant cost in terms of performance.

### Antialiasing
Antialiasing is a technique for smoothing an image by taking multiple samples at different locations per pixel. Instead of firing one ray at the center of its assigned pixel, we subdivide the pixel into equal cells and fire a ray at the center of each of those cells. Finally, when coloring the image, we average the colors assigned to each of the cells in a pixel. The result is as follows:

**TODO**

Clearly, the depth of field technique especially benefits from the use of antialiasing. One of the drawbacks of depth of field is that the runtime and memory usage scales linearly with the number of samples per pixel. The impact on performance is indicated by the following graphic:

**TODO**

<b id="f1">1</b> The distinction between iterations and time-steps may be a little confusing. Within a time-step, a light ray bounces (at most) once -- it moves from one surface to another or strikes empty space. In contrast, an iteration is only complete once all rays have terminated. This generally involves multiple time-steps and bounces. The purpose of an iteration is to denoise an image by averaging over multiple possible random light paths. [↩](#1)

<b id="f2">2</b> Technically the angle is defined by the _ratio_ of the refraction indices of the substances involved, e.g. air to water if the ray is entering water from the air. [↩](#2)
