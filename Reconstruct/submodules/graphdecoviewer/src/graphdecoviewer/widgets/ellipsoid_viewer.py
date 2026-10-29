import numpy as np
from . import Widget
from OpenGL.GL import *
from .cameras import Camera
from ..types import Texture2D, CLIENT
from imgui_bundle import imgui
from OpenGL.GL.shaders import compileShader, compileProgram

_vert_shader = """
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */


#version 430

uniform mat4 MVP;
uniform float alpha_limit;
uniform int stage;
uniform float scaling_modifier;

layout (std430, binding = 0) buffer BoxCenters {
    float centers[];
};
layout (std430, binding = 1) buffer Rotations {
    vec4 rots[];
};
layout (std430, binding = 2) buffer Scales {
    float scales[];
};
layout (std430, binding = 3) buffer Alphas {
    float alphas[];
};
layout (std430, binding = 4) buffer Colors {
    float colors[];
};

mat3 quatToMat3(vec4 q) {
  float qx = q.y;
  float qy = q.z;
  float qz = q.w;
  float qw = q.x;

  float qxx = qx * qx;
  float qyy = qy * qy;
  float qzz = qz * qz;
  float qxz = qx * qz;
  float qxy = qx * qy;
  float qyw = qy * qw;
  float qzw = qz * qw;
  float qyz = qy * qz;
  float qxw = qx * qw;

  return mat3(
    vec3(1.0 - 2.0 * (qyy + qzz), 2.0 * (qxy - qzw), 2.0 * (qxz + qyw)),
    vec3(2.0 * (qxy + qzw), 1.0 - 2.0 * (qxx + qzz), 2.0 * (qyz - qxw)),
    vec3(2.0 * (qxz - qyw), 2.0 * (qyz + qxw), 1.0 - 2.0 * (qxx + qyy))
  );
}

const vec3 boxVertices[8] = vec3[8](
    vec3(-1, -1, -1),
    vec3(-1, -1,  1),
    vec3(-1,  1, -1),
    vec3(-1,  1,  1),
    vec3( 1, -1, -1),
    vec3( 1, -1,  1),
    vec3( 1,  1, -1),
    vec3( 1,  1,  1)
);

const int boxIndices[36] = int[36](
    0, 1, 2, 1, 3, 2,
    4, 6, 5, 5, 6, 7,
    0, 2, 4, 4, 2, 6,
    1, 5, 3, 5, 7, 3,
    0, 4, 1, 4, 5, 1,
    2, 3, 6, 3, 7, 6
);

out vec3 worldPos;
out vec3 ellipsoidCenter;
out vec3 ellipsoidScale;
out mat3 ellipsoidRotation;
out vec3 colorVert;
out float alphaVert;
out flat int boxID;

void main() {
	boxID = gl_InstanceID;
    float a = alphas[boxID];

    // Early exit
    if ((stage == 0 && a < alpha_limit) || (stage == 1 && a >= alpha_limit)) {
        gl_Position = vec4(0,0,0,0);
        return;
    }

    ellipsoidCenter = vec3(centers[3 * boxID + 0], centers[3 * boxID + 1], centers[3 * boxID + 2]);
	alphaVert = a;
	ellipsoidScale = vec3(scales[3 * boxID + 0], scales[3 * boxID + 1], scales[3 * boxID + 2]);
	ellipsoidScale = 2 * ellipsoidScale * scaling_modifier;

	vec4 q = rots[boxID];
	ellipsoidRotation = transpose(quatToMat3(q));

    int vertexIndex = boxIndices[gl_VertexID];
    worldPos = ellipsoidRotation * (ellipsoidScale * boxVertices[vertexIndex]);
    worldPos += ellipsoidCenter;

	float r = colors[boxID * 48 + 0] * 0.2 + 0.5;
	float g = colors[boxID * 48 + 1] * 0.2 + 0.5;
	float b = colors[boxID * 48 + 2] * 0.2 + 0.5;

	colorVert = vec3(r, g, b);
	
    gl_Position = MVP * vec4(worldPos, 1);
}
"""

_frag_shader = """
/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */

#version 430

uniform mat4 MVP;
uniform float alpha_limit;
uniform int stage;
uniform vec3 rayOrigin;

in vec3 worldPos;
in vec3 ellipsoidCenter;
in vec3 ellipsoidScale;
in mat3 ellipsoidRotation;
in vec3 colorVert;
in float alphaVert;
in flat int boxID;

layout (location = 0) out vec4 out_color;
// layout (location = 1) out uint out_id;

vec3 closestEllipsoidIntersection(vec3 rayDirection, out vec3 normal) {
  // Convert ray to ellipsoid space
  dvec3 localRayOrigin = (rayOrigin - ellipsoidCenter) * ellipsoidRotation;
  dvec3 localRayDirection = normalize(rayDirection * ellipsoidRotation);

  dvec3 oneover = double(1) / dvec3(ellipsoidScale);
  
  // Compute coefficients of quadratic equation
  double a = dot(localRayDirection * oneover, localRayDirection * oneover);
  double b = 2.0 * dot(localRayDirection * oneover, localRayOrigin * oneover);
  double c = dot(localRayOrigin * oneover, localRayOrigin * oneover) - 1.0;
  
  // Compute discriminant
  double discriminant = b * b - 4.0 * a * c;
  
  // If discriminant is negative, there is no intersection
  if (discriminant < 0.0) {
    return vec3(0.0);
  }
  
  // Compute two possible solutions for t
  float t1 = float((-b - sqrt(discriminant)) / (2.0 * a));
  float t2 = float((-b + sqrt(discriminant)) / (2.0 * a));
  
  // Take the smaller positive solution as the closest intersection
  float t = min(t1, t2);
  
  // Compute intersection point in ellipsoid space
  vec3 localIntersection = vec3(localRayOrigin + t * localRayDirection);

  // Compute normal vector in ellipsoid space
  vec3 localNormal = normalize(localIntersection / ellipsoidScale);
  
  // Convert normal vector to world space
  normal = normalize(ellipsoidRotation * localNormal);
  
  // Convert intersection point back to world space
  vec3 intersection = ellipsoidRotation * localIntersection + ellipsoidCenter;
  
  return intersection;
}

void main(void) {
	vec3 dir = normalize(worldPos - rayOrigin);

	vec3 normal;
	vec3 intersection = closestEllipsoidIntersection(dir, normal);
	float align = max(0.4, dot(-dir, normal));
	
	out_color = vec4(1, 0, 0, 1);
	
	if(intersection == vec3(0))
		discard;

	vec4 newPos = MVP * vec4(intersection, 1);
	newPos /= newPos.w;

	gl_FragDepth = newPos.z;

	float a = stage == 0 ? 1.0 : 0.05f;

	out_color = vec4(align * colorVert, a);
// 	out_id = boxID;
}
"""

class EllipsoidViewer(Widget):
    def __init__(self, mode):
        super().__init__(mode)
        self.limit = 0.2
        self.scaling_modifier = 1
        self.render_floaters = False
        self.num_gaussians = None
        self.step_called = False
        self.enabled = False

    def setup(self):
        """ Create the buffers for storing the gaussian parameters and the framebuffers to render to. """
        self._color_texture = Texture2D()
        self._color_texture.id = glGenTextures(1)
        self._depth_texture = Texture2D() # Technically its a RBO
        self._depth_texture.id = glGenRenderbuffers(1)
        self._fbo = None

        # Create buffers for Gaussian Attributes
        self._means = glGenBuffers(1)
        self._rotations = glGenBuffers(1)
        self._scales = glGenBuffers(1)
        self._alphas = glGenBuffers(1)
        self._colors = glGenBuffers(1)

        try:
            # Create shaders
            self._shader = compileProgram(
                compileShader(_vert_shader, GL_VERTEX_SHADER),
                compileShader(_frag_shader, GL_FRAGMENT_SHADER),
            )

            self._vao = glGenVertexArrays(1)
            glBindVertexArray(self._vao)

            # Create a query for timing
            self.query = glGenQueries(1)[0]

            # Create a dummy FBO because `step` is not called in CLIENT mode
            if self.mode is CLIENT:
                self._create_fbo(1, 1)

            self.enabled = True
        except Exception as e:
            print(f"Error setting up EllipsoidViewer: {e}")

    def destroy(self):
        glDeleteTextures(1, int(self._color_texture.id))
        glDeleteRenderbuffers(1, int(self._depth_texture.id))
        glDeleteBuffers(1, int(self._means))
        glDeleteBuffers(1, int(self._rotations))
        glDeleteBuffers(1, int(self._scales))
        glDeleteBuffers(1, int(self._alphas))
        glDeleteBuffers(1, int(self._colors))
        glDeleteQueries(1, int(self.query))
        if self._fbo is not None:
            glDeleteFramebuffers(1, int(self._fbo))
        glDeleteProgram(self._shader)

    def _create_fbo(self, res_x: int, res_y: int):
        # Create framebuffer
        if self._fbo is not None:
            glDeleteFramebuffers(self._fbo)
        self._fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)

        # Create texture to render to
        self._color_texture.res_x = res_x
        self._color_texture.res_y = res_y
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA8,
            self._color_texture.res_x, self._color_texture.res_y,
            0, GL_RGBA, GL_UNSIGNED_BYTE, None
        )
        glBindTexture(GL_TEXTURE_2D, 0)
        # Attach texture to framebuffer
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            self._color_texture.id, 0
        )

        # Create depth RBO
        self._depth_texture.res_x = res_x
        self._depth_texture.res_y = res_y
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_texture.id)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8,
            self._depth_texture.res_x, self._depth_texture.res_y
        )
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        # Attach it framebuffer
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,
            self._depth_texture.id
        )

        # Verify framebuffer is complete
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "FBO not complete"

        # Unbind the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def upload(self, means: np.ndarray, rotations: np.ndarray, scales: np.ndarray, alphas: np.ndarray, colors: np.ndarray):
        """ Upload gaussian parameters to OpenGL buffers. """
        self.num_gaussians = means.shape[0]
        glBindBuffer(GL_ARRAY_BUFFER, self._means)
        glBufferData(GL_ARRAY_BUFFER, means.nbytes, means, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._rotations)
        glBufferData(GL_ARRAY_BUFFER, rotations.nbytes, rotations, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._scales)
        glBufferData(GL_ARRAY_BUFFER, scales.nbytes, scales, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._alphas)
        glBufferData(GL_ARRAY_BUFFER, alphas.nbytes, alphas, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._colors)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def step(self, camera: Camera):
        if not self.enabled:
            return
        
        glBeginQuery(GL_TIME_ELAPSED, self.query)
        if self._color_texture.res_x != camera.res_x or \
            self._color_texture.res_y != camera.res_y:
            self._create_fbo(camera.res_x, camera.res_y)
        
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self._means)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._rotations)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._scales)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._alphas)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._colors)

        # TODO: Check why this is needed
        # // Solid pass
		# GLuint drawBuffers[2];
		# drawBuffers[0] = GL_COLOR_ATTACHMENT0;
		# drawBuffers[1] = GL_COLOR_ATTACHMENT1;
        # glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

        # Use the shader program
        glUseProgram(self._shader)

        glUniformMatrix4fv(glGetUniformLocation(self._shader, "MVP"), 1, GL_TRUE, camera.full_projection)
        glUniform3fv(glGetUniformLocation(self._shader, "rayOrigin"), 1, camera.origin)
        glUniform1f(glGetUniformLocation(self._shader, "alpha_limit"), float(self.limit))
        glUniform1i(glGetUniformLocation(self._shader, "stage"), 0)
        glUniform1f(glGetUniformLocation(self._shader, "scaling_modifier"), float(self.scaling_modifier))
        glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_gaussians)

        if self.render_floaters:
            glDepthMask(GL_FALSE)
            glEnable(GL_BLEND);
            glBlendEquation(GL_FUNC_ADD)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glUniform1i(glGetUniformLocation(self._shader, "stage"), 1)
            glDrawArraysInstanced(GL_TRIANGLES, 0, 36, self.num_gaussians)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        # Unbind program
        glUseProgram(0)

        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Measure time required for rendering
        glEndQuery(GL_TIME_ELAPSED)

        self.step_called = True

    def server_send(self):
        if not self.step_called:
            return None, None
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        arr = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.step_called = False
        return arr, {"shape": (self._color_texture.res_y, self._color_texture.res_x, 3)}
    
    def client_recv(self, binary, text):
        img = np.frombuffer(binary, dtype=np.uint8).reshape(text["shape"])
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
        # exit()
        res_y = text["shape"][0]
        res_x = text["shape"][1]
        # img = binary
        glBindTexture(GL_TEXTURE_2D, self._color_texture.id)
        if self._color_texture.res_x != res_x  or self._color_texture.res_y != res_y:
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, res_x, res_y, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            self._color_texture.res_x = res_x
            self._color_texture.res_y = res_y
        else:
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, res_x, res_y, GL_RGB, GL_UNSIGNED_BYTE, img)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def show_gui(self, draw_list: imgui.ImDrawList=None):
        res_x = self._color_texture.res_x
        res_y = self._color_texture.res_y
        if draw_list is not None:
            draw_list.add_image(self._color_texture.tex_ref, (0, 0), (res_x, res_y))
        else:
            imgui.image(self._color_texture.tex_ref, (res_x, res_y))