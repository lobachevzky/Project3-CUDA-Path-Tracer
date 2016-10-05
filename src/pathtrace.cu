#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define sortByMaterial 0
#define cache1stBounce 0
#define raysPerPixelAxis 2
#define raysPerPixel (raysPerPixelAxis * raysPerPixelAxis)
#define camJitter 0.3f
#define depthOfField 8.0f

#define DEBUG 1
#define lightIdx 170050
#define refractIdx 310062
#define printIters 2
#define N 5

#if DEBUG
#define debug(...) if (iter < printIters) { printf(__VA_ARGS__); }
#define debugN(...) if (index == N && iter < printIters) { printf(__VA_ARGS__); }
#define debugLight(...) if (index == lightIdx && iter < printIters) { printf(__VA_ARGS__); }
#define debugRefract(...) if (index == refractIdx && iter < printIters) { printf(__VA_ARGS__); }
#else
#define debug(...) {}
#define debugN(...) {}
#define debugLight(...) {}
#define debugRefract(...) {}
#endif


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static PathSegment *dev_1stPaths = NULL;
static ShadeableIntersection *dev_1stIntersects = NULL;
static glm::vec3 *dev_colors = NULL;
static ShadeableIntersection *dev_intersects = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;
	const int num_rays = pixelcount * raysPerPixel;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, num_rays * sizeof(PathSegment));
	cudaMalloc(&dev_1stPaths, num_rays * sizeof(PathSegment));

	cudaMalloc(&dev_1stIntersects, num_rays * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_colors, num_rays * sizeof(glm::vec3));
	cudaMemset(dev_colors, 0, num_rays * sizeof(glm::vec3));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersects, num_rays * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersects, 0, num_rays * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_1stPaths);
  	cudaFree(dev_colors);
  	cudaFree(dev_1stIntersects);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersects);
    // TODO: clean up any extra device memory you created

    checkCUDAError("pathtraceFree");
}

/**
	{

	};
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(
	const Camera cam, 
	const int iter, 
	const int traceDepth,
	PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int raysXAxis = cam.resolution.x * raysPerPixelAxis;
  int raysYAxis = cam.resolution.y * raysPerPixelAxis;

	if (x < raysXAxis && y < raysYAxis) {

    int pixelIdx = (cam.resolution.x * y / raysPerPixelAxis) + (x / raysPerPixelAxis);

    int pixelOffset_x = x % raysPerPixelAxis;
    int pixelOffset_y = y % raysPerPixelAxis;

    int index = (pixelIdx * raysPerPixel) 
      + (raysPerPixelAxis * pixelOffset_y) + pixelOffset_x;

    //debugN("pixel_x: %d, pixel_y: %d, pixelIdx: %d\n", pixel_x, pixel_y, pixelIdx);
    //debugN("rayX: %d, rayY: %d\n", pixelOffset_x, pixelOffset_y);
    //debugN("x: %d, y: %d\n", x, y);

		PathSegment & segment = pathSegments[index];

		thrust::default_random_engine rng = makeSeededRandomEngine(x, y, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		float jx = u01(rng) * camJitter;
		float jy = u01(rng) * camJitter;
		float jz = u01(rng) * camJitter;
		glm::vec3 jitter(jx, jy, 0.0f);

		segment.ray.origin = cam.position + jitter;
		segment.color = glm::vec3(1.0f);

		// TODO: implement antialiasing by jittering the ray

		glm::vec3 pixel_dir = cam.view
			+ cam.right * cam.pixelLength.x / (float)raysPerPixelAxis
			* ((float)cam.resolution.x * raysPerPixelAxis * 0.5f - (float)x)
			+ cam.up * cam.pixelLength.y / (float)raysPerPixelAxis
			* ((float)cam.resolution.y * raysPerPixelAxis * 0.5f - (float)y);

		glm::vec3 pixel = cam.position + glm::normalize(pixel_dir) * depthOfField;

		segment.ray.direction = glm::normalize(pixel - segment.ray.origin);
		segment.colorIndex = index;
		segment.remainingBounces = traceDepth;
		segment.ray.insideObject = 0;
	}
}

__global__ void pathTraceOneBounce(
	  const int num_paths
	, const PathSegment * pathSegments
	, const Geom * geoms
	, const int geoms_size
	, ShadeableIntersection * intersections
	, const Material * materialArray
	, const int iter
	, Camera cam
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths) return;

	PathSegment segment = pathSegments[path_index];

	float distance_from_origin;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float nearest_distance = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms

  for (int i = 0; i < geoms_size; i++)
  {
    Geom geom = geoms[i];

    if (geom.type == CUBE)
    {
      distance_from_origin = boxIntersectionTest(geom, segment.ray, tmp_intersect, tmp_normal, outside);
    }
    else if (geom.type == SPHERE)
    {
      distance_from_origin = sphereIntersectionTest(geom, segment.ray, tmp_intersect, tmp_normal, outside);
    }
    // TODO: add more intersection tests here... triangle? meatball? CSG?

    // Compute the minimum t from the intersection tests to determine what
    // scene geometry object was hit first.

    if (distance_from_origin > 0.0f && nearest_distance > distance_from_origin)
    {
      nearest_distance = distance_from_origin;
      hit_geom_index = i;
      intersect_point = tmp_intersect;
      normal = tmp_normal;
    }

    glm::vec3 o = segment.ray.origin;
  }

  ShadeableIntersection &intersection = intersections[path_index];
	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
	}
	else
	{
		//The ray hits something
		intersection.t = nearest_distance;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = normal;
		intersection.point = intersect_point;
	}
}

void __device__ finalize(PathSegment &segment, glm::vec3 *colors) {
	segment.remainingBounces = 0;
	colors[segment.colorIndex] = segment.color;
}

void __device__ decrementBounces(PathSegment &segment, glm::vec3 *colors) {
	segment.remainingBounces--;
	if (segment.remainingBounces == 0) {
		finalize(segment, colors);
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection *shadeableIntersections
	, PathSegment *pathSegments
	, glm::vec3 *colors
	, Material *materials
	, int i
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  PathSegment &segment = pathSegments[idx];

  int index = idx;

	if (idx >= num_paths) return;

	assert(segment.remainingBounces > 0);

	ShadeableIntersection intersection = shadeableIntersections[idx];
	if (intersection.t > 0.0f) { // if the intersection exists...
		Material material = materials[intersection.materialId];
		segment.color *= material.color;

    debugRefract("Shaded\n");

		// If the material indicates that the object was a light, "light" the ray
		if (material.emittance > 0.0f) {
			segment.color *= material.emittance;
			finalize(segment, colors);
			glm::vec3 l = intersection.point;
      debugRefract("Hit Light\n");
		}
		// Otherwise, rescatter.
		else {
			/*__host__ __device__
				thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
				int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
				return thrust::default_random_engine(h);
			}*/
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

			scatterRay(segment.ray, intersection, material, rng);
			decrementBounces(segment, colors);
      if (segment.remainingBounces == 0) {
        debugRefract("Out of Bounces\n");
      }

		} // If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
	} else { 
		segment.color *= 0; 
		finalize(segment, colors);
		glm::vec3 c = segment.color;
    debugRefract("No Hit\n");
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int pixelcount, int resolutionX,
	glm::vec3 *image, glm::vec3 *colors)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < pixelcount)
	{
		//PathSegment iterationPath = iterationPaths[index];
		glm::vec3 newColor(0.0f);
		for (int i = 0; i < raysPerPixel; i++) {
			newColor += colors[idx * raysPerPixel + i];
		}
		image[idx] += newColor / (float)raysPerPixel;
	}
}

struct materialType
{
  __host__ __device__
	  bool operator()(const ShadeableIntersection s1, const ShadeableIntersection s2)
  {
	  return s1.materialId > s2.materialId;
  }
};
struct noMoreBounces
{
  __host__ __device__
	  bool operator()(const PathSegment s)
  {
	  return (s.remainingBounces == 0);
  }
};


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
					(cam.resolution.x * raysPerPixelAxis + blockSize2d.x - 1) / blockSize2d.x,
					(cam.resolution.y * raysPerPixelAxis + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	//if (iter > 0) return;
	//glm::vec3 v(1.0, 0.0, 0.0);
	//glm::vec3 n(-1.0, -1.0, 0.0);
	//glm::vec3 r = glm::refract(v, n, 0.5f); 
	//debug("%f %f %f\n", r.x, r.y, r.z);
	//r = glm::refract(r, n, 10000.0f);
	//debug("%f %f %f\n", r.x, r.y, r.z);
	//return;



  int num_paths = pixelcount * raysPerPixel;
	if (iter == 0 || !cache1stBounce) {
		checkCUDAError("before generate camera ray");
		generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");
		cudaMemcpy(
			dev_1stPaths, dev_paths, 
			num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		checkCUDAError("cudaMemcpy dev_paths to dev_1stPaths");
	}
	else {
		cudaMemcpy(
			dev_paths, dev_1stPaths, 
			num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
	}


	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	int i = 0;
	bool iterationComplete = false;
	while (!iterationComplete) {
		checkCUDAError("generate camera ray");
		// clean shading chunks
		cudaMemset(dev_intersects, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d; 
		if (iter == 0 || i > 0 || !cache1stBounce) {
			pathTraceOneBounce <<<numblocksPathSegmentTracing, blockSize1d>>> (
					num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersects
				, dev_materials
				, iter
				, cam
			); 
			checkCUDAError("trace one bounce");
		}

		if (iter == 0 && i == 0 && cache1stBounce) {
			cudaMemcpy(
				dev_1stIntersects, dev_intersects, 
				num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			checkCUDAError("memCpy to dev_1stIntersects");
		}
		if (iter > 0 && i == 0 && cache1stBounce) {
			cudaMemcpy(
				dev_intersects, dev_1stIntersects, 
				num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			checkCUDAError("memCpy to dev_intersects");
		}
		cudaDeviceSynchronize();

		// TODO
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (sortByMaterial) {
			thrust::device_ptr<ShadeableIntersection> thrust_intersects(dev_intersects);
			thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
			thrust::sort_by_key(
				thrust_intersects,
				thrust_intersects + num_paths,
				thrust_paths,
				materialType()
				);
		}

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersects,
			dev_paths,
			dev_colors,
			dev_materials,
			i
		);
		checkCUDAError("after shading");

		thrust::device_ptr<PathSegment> thrust_paths(dev_paths);
		thrust::device_ptr<PathSegment> end_paths;

		end_paths = thrust::remove_if(
			thrust_paths, thrust_paths + num_paths, noMoreBounces()
		);
		num_paths = end_paths - thrust_paths;
		checkCUDAError("after stream compaction");
		if (num_paths == 0) {
			iterationComplete = true; // TODO: should be based off stream compaction results. 
		} 
		i++;
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
  finalGather << <numBlocksPixels, blockSize1d >> >(pixelcount, cam.resolution.x, dev_image, dev_colors);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
					pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
