CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/aa3dof.png)

## Summary
For this project, I implemented part of a pathtracer, a program for rendering an image of a scene, given the locations and attributes of shapes in 3D space. Unlike a physical camera, which perceives objects by capturing light that has bounced off of them from a light source, a pathtracer follows the path of light in reverse: on each iteration, the program projects rays from the camera toward each pixel on the screen. With each time-step<sup id="1">[1](#f1)</sup>, the ray bounces off a surface and samples colors from it. The ray terminates if it strikes a light source, runs out of bounces, or strikes empty space.

## Basic Features
### Scatter mechanism
Once a ray strikes an object (besides a light source), it bounces and a new path (origin and direction) must be calculated. My scatter function handles multiple cases:
- When a material is fully reflective (ideal specular). This is the simplest of the cases: the ray is simply reflected with the angle of incidence equal to the angle of reflection.
- When a material is fully diffuse, the ray bounces off at a completely random angle within the hemisphere defined by the surface normal. For example, if a ray bounces off an ordinary wall surface, it's new path is in a random direction, excluding paths that actually penetrate the wall.
- When a material is diffractive, light penetrates the surface but bends based on the refraction index of the material. <sup id="2">[2](#f2)</sup>
- When a material is both refractive and reflective, the ray chooses randomly between refraction and reflection, using a distribution defined by the properties of the material (its ratio of "hasRefractive" to "hasReflective").

### Shading mechanism
A ray begins as white and as it strikes a material, multiplies its current color by the material's color. If the ray strikes a light source, the accumulated color value is multiplied by the brightness of the light. If it strikes empty space or runs out of bounces, the color is set to black. This effect accounts for shadows, because surfaces that do not have a direct path to a light source have a lower chance of reflecting a ray into it.

## Optimizations
### Ray compaction
Every time-step, rays may terminate by striking empty space or a light. A naive approach to handling these rays would be to set a flag indicating that they are no longer active and then check this flag at the start of the shading kernel to prevent further coloration. The problem with this approach is that the threads assigned to these dead rays would be _wasted_. Instead, we perform stream compaction on the rays at the end of every time step to eliminate dead rays.

A pitfall of this optimization (one which cost me many hours of debugging) is that stream-compaction mutates the compacted array. Consequently, dead rays must be saved somehow so that their colors can be rendered at the end of the iteration.

One naive approach is to make a second array of pointers to the array of rays. Then we perform all operations, including stream compaction on the array of pointers instead of the array of rays. When we perform stream compaction, we only eliminate the pointers, not the rays themselves. Finally, once all pointers have been eliminated, we use the original array to render the image. This approach is depicted in this graphic:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/ray-pointers/ray-pointers.001.png)

A more performant approach is to maintain a separate array of color values in addition to the array of rays. Whenever we terminate a ray, we first store its color in the color array. Finally we use the color array to render the final image. This approach is depicted here:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/ray-pointers/ray-pointers.002.png)

In the naive approach, every manipulation of rays involves following pointers through global memory. Specifically, for each ray, the naive approach retrieves the pointer from global memory and then follows that pointer to another address in global memory. These global memory accesses are extremely slow. The second approach accesses the rays directly in global memory instead of following pointers (one memory access instead of two) and only performs a second memory access to the color array if the ray terminates.

### Storing materials in contiguous memory
When a ray strikes a surface, we must access that surface's material from global memory. If the same block accesses different materials in memory, then it is likely that some accesses will take longer than others. However, the entire block will have to wait on the slowest access. By sorting rays by material type, we can increase the chances that a block handles a single material and that all access times will be the same. This should increase hardware saturation. In order to achieve this, we used `thrust::sort_by_key` to sort the rays by the materials associated with their corresponding surface intersections. Unfortunately, we found that this did not improve performance at all: both programs performed with exactly 62.5% occupancy on the `shadeMaterial` function (the impacted function).

### Caching the first bounce
In a typical pathtracer, all rays follow the same path on the first bounce: from the camera to their assigned pixel. Consequently it is unnecessary to recalculate this first bounce every time. If the `cache1stBounce` flag is set to 1, then the program caches the first segment in `dev_1stpath` and the first intersection in `dev_1stIntersect`. On average, `generateRayFromCamera` takes 613.28 microseconds while `pathTraceOneBounce` takes a whopping 17,178.944 microseconds (17 milliseconds). By caching the first call to both of these functions, the program saves 17,792.224 microseconds (18 milliseconds) every iteration, almost 1/8th the runtime of an entire iteration. Here is a chart demonstrating the performance difference:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/performance%20profiles/profiles_Page_1.png)

## Extra Features
### Refraction
As mentioned in the section on the scatter mechanism, the program implements refraction in addition to reflection and diffuse scattering. The program only handles cases where light enters a refractive material from air or enters air from a refractive material. When the ray enters the refractive material, a toggle in the ray struct is set to `1`. If the ray strikes a refractive material from inside an object (as indicated by the toggle), the ratio of the indices of refraction is inverted -- this causes the light to bend back toward it's original direction as depicted in this image:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/light-refraction-glass.gif)

**Performance:** Refraction is not a performance-intensive feature. There is nothing GPU-specific about this feature -- the feature would not perform any differently on a CPU. One way, however, that ray-scattering might generally be optimized is by storing `ShadeableIntersection` structs, which store information about the point where a ray intersects a surface, and `PathSegment` structs, which store information about the segment of a ray associated with a single bounce, in shared memory each time-step. This way,  subsequent accesses of these structs do not require calls to global memory. This is not difficult to implement since for the duration of a time-step, the arrays containing these structs are not reshuffled at all.

**Optimization:** With the current implementation, the image appears a looks to clean and a little artificial. A more effective implementation might include elements of subsurface scattering or at the least, it might benefit from random mixes of refraction, reflection, and diffusion.

### Depth of field
Because a lens can precisely focus at only one distance at a time, objects at different distances may appear out of focus. In order to implement this feature, we jittered the camera by applying a random, small offset to its position and then recalculating the direction of the ray from its new origin to its assigned pixel (not recalculating the direction just causes the entire image to become blurry). Here is a comparison of the image, with and without depth of field added:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/blurNoBlurComparison.png)

**Performance:** Depth-of-field does not inherently benefit from the use of a GPU or have any performance costs. However, I was curious whether it would not be compatible with the first-bounce-caching optimization. Without caching, the cameras is set to a new, random starting position on each iteration, whereas with caching, the camera always starts from the same random offset. This is the result of using depth-of-field with caching:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/cachingNoCachingComparison.png)
The image on the right uses the caching optimization whereas the image on the left does not.

**Optimization:** Even with antialiasing, out-of-focus edges appear ragged and noisy instead of looking blurred, as they should. One solution is to use some kind of convolution to blur noisy areas.

One nice feature of depth-of-field is that it has absolutely no impact on performance, although images employing depth-of-field benefit significantly from antialiasing, which does come at a significant cost in terms of performance.

### Antialiasing
Antialiasing is a technique for smoothing an image by taking multiple samples at different locations per pixel. Instead of firing one ray at the center of its assigned pixel, we subdivide the pixel into equal cells and fire a ray at the center of each of those cells. Finally, when coloring the image, we average the colors assigned to each of the cells in a pixel. The result is as follows:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/AANoAAComparison.png)
The image on the right employs antialiasing x9 whereas the image on the left does not. The difference is evident in the amount of noise in the left image.

**Performance** One of the drawbacks of depth of field is that the number of threads and memory usage scales linearly with the number of samples per pixel. The GPU somewhat mitigates this because, on a CPU, none (or few) of these additional threads could be run in parallel. Here is a chart comparing levels of antialiasing:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/performance%20profiles/profiles_Page_3.png)

**Optimization** Instead of having a color array whose length is a multiple of the number of pixels, it would be possible for the color array to have as many elements as pixels if we averaged the colors in place. For example, in the current system, with antialiasing x4, the indices `[0, 1, 2, 3]` in the color array are assigned to pixel 0. Separate colors are assigned to each of these indices and then averaged in the "final gather" step (in which colors are actually assigned to `dev_image`, the image object). Instead, we could simply add the colors together in index 0 as their corresponding rays terminate.

One drawback of this approach is that multiple threads would be writing to the same global address in memory (index 0 in our example). Consequently, some kind of synchronization would be required to prevent race conditions. This is no slower than the current approach, which synchronously adds the separate colors in the final gather step.

Another space optimization would be to eliminate the "final gather" step by writing directly to the `dev_image` array. This would be possible given the previous optimization.       

### Specular noise
In reality, light rays do not bounce perfectly off of specular surfaces. Instead, they reflect randomly within a cone centered on the perfect reflection. This zoomed in image shows the result of specular noise:

![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/img/reflectionZoom.png)

**Performance** Specular noise has no effect on performance. The only impact is that some additional computation needs to be performed on reflective rays.

**Optimization** Ideally specular noise would vary across a surface to reflect the mottled character of most physical surfaces.

### BlockSize analysis
Here is a chart comparing performance across block sizes.
![alt text] (https://github.com/lobachevzky/Project3-CUDA-Path-Tracer/blob/master/performance%20profiles/profiles_Page_2.png)

It should be noted that these comparisons were done without antialiasing. With significant antialiasing, all block sizes threw an error except 128 and 256 and of these two, 128 was faster.

<b id="f1">1</b> The distinction between iterations and time-steps may be a little confusing. Within a time-step, a light ray bounces (at most) once -- it moves from one surface to another or strikes empty space. In contrast, an iteration is only complete once all rays have terminated. This generally involves multiple time-steps and bounces. The purpose of an iteration is to de-noise an image by averaging over multiple possible random light paths. [↩](#1)

<b id="f2">2</b> Technically the angle is defined by the _ratio_ of the refraction indices of the substances involved, e.g. air to water if the ray is entering water from the air. [↩](#2)
