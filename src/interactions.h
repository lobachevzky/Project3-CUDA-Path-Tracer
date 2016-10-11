#pragma once

#include "intersections.h"
#define DEBUG 1
#define lightIdx 170050
#define refractIdx 310062
#define printIters 2
#define specularNoise 0.05f

#if DEBUG
#define debug(...) if (iter < printIters) { printf(__VA_ARGS__); }
#define debugLight(...) if (index == lightIdx && iter < printIters) { printf(__VA_ARGS__); }
#define debugRefract(...) if (index == refractIdx && iter < printIters) { printf(__VA_ARGS__); }
#else
#define debug(...) {}
#define debugLight(...) {}
#define debugRefract(...) {}
#endif

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic a.lso applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
        Ray &ray,
		ShadeableIntersection intersection,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	int reflect, refract;
	if (m.hasReflective > 0.0 && m.hasRefractive > 0) {
		float ratio = m.hasRefractive / m.hasReflective;
		refract = exp(ratio)/(1 + exp(-ratio)) > u01(rng);
		reflect = !refract;
	}
	else {
		reflect = m.hasReflective > 0.0;
		refract = m.hasRefractive > 0.0;
	}
	if (reflect) {
		glm::vec3 reflection = glm::reflect(ray.direction, intersection.surfaceNormal);

		float jx = u01(rng) * 1.0;
		float jy = u01(rng) * 1.0;
		float jz = u01(rng) * 1.0;

		glm::vec3 rand_normal(jx, jy, jz);
		ray.direction = glm::rotate(reflection, specularNoise, rand_normal);
	}
	if (refract) { 
		assert(m.hasRefractive > 0.0f);
		ray.insideObject = !ray.insideObject;
		float eta = ray.insideObject ? 1.0 / m.indexOfRefraction : m.indexOfRefraction;
		ray.direction = glm::refract(ray.direction, intersection.surfaceNormal, eta);
	}
	if (!reflect && !refract) {
		ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
	} 
	ray.origin = intersection.point + (ray.direction * EPSILON);
}
