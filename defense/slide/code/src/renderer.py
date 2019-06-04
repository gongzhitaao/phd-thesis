from types import SimpleNamespace

from objloader import Obj
from PIL import Image
from pyrr import matrix44
import moderngl
import numpy as np

# Average ImageNet pixel.
# (R, G, B) = (0.485, 0.456, 0.406)
(R, G, B) = (1, 1, 1)

ctx = moderngl.create_standalone_context()

(WIDTH, HEIGHT) = (800, 800)
fbo = ctx.simple_framebuffer((WIDTH, HEIGHT))


class MGLRenderer:
    vert_shader = """
#version 330

uniform bool is_background;
uniform mat4 Mvp;

in vec3 in_vert;
in vec3 in_norm;
in vec2 in_text;

out vec3 v_norm;
out vec2 v_text;

void main() {
  if (!is_background) {
      gl_Position = Mvp * vec4(in_vert, 1.0);
      v_norm = in_norm;
      v_text = in_text;
  } else {
      gl_Position = vec4(in_vert, 1.0);
      v_norm = in_norm;
      v_text = in_text;
  }
}
    """

    frag_shader = """
#version 330

uniform vec3 DirLight;
uniform float dir_int;
uniform float amb_int;
uniform sampler2D Texture;
uniform bool is_background;

in vec3 v_norm;
in vec2 v_text;

out vec4 f_color;

void main() {
  if (!is_background) {
      float lum = clamp(dot(DirLight, v_norm), 0.0, 1.0) * dir_int + amb_int;
      f_color = vec4(texture(Texture, v_text).rgb * lum,
                     texture(Texture, v_text).a);
  } else {
      f_color = vec4(texture(Texture, v_text).rgba);
  }
}
    """

    def __init__(self):
        self.bufs = {}

        self.ctx = ctx
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)

    def release(self):
        for _, v in self.bufs.items():
            v.release()

    def loadobj(self, objfn, texfn):
        self.release()

        prog = self.ctx.program(
            vertex_shader=self.vert_shader, fragment_shader=self.frag_shader)

        prog["is_background"].value = False
        prog["DirLight"].value = (-1, 1, 1)
        prog["dir_int"].value = 0.4
        prog["amb_int"].value = 1.0
        self.bufs['prog'] = prog

        self.vs = SimpleNamespace()
        self.vs.proj = matrix44.create_perspective_projection(
            30, 1.0, 0.1, 1000.0)
        self.vs.view = matrix44.create_look_at(
            [0, 0, 1.5],
            [0, 0, 0],
            [0, 1, 0],
        )

        self.bufs['fbo'] = fbo
        fbo.use()

        obj = Obj.open(objfn)
        tmp = obj.pack('vx vy vz nx ny nz tx ty')
        arr = np.array(np.frombuffer(tmp, dtype='f4')).reshape((-1, 8))

        # move obj center to the origin
        tmp = arr[:, :3]
        center = np.mean(tmp, axis=0)
        tmp -= center

        # scale obj to be within [-1, 1]
        a, b = tmp.min(), tmp.max()
        arr[:, :3] = tmp / (b - a)

        vbo = self.ctx.buffer(arr.flatten().astype("f4").tobytes())
        vao = self.ctx.simple_vertex_array(self.bufs['prog'], vbo, "in_vert",
                                           "in_norm", "in_text")
        self.bufs['vbo'] = vbo
        self.bufs['vao'] = vao

        img = Image.open(texfn).transpose(
            Image.FLIP_TOP_BOTTOM).convert("RGBA")
        texture = self.ctx.texture(img.size, 4, img.tobytes())
        texture.build_mipmaps()
        texture.use()
        self.bufs['texture'] = texture

    def render(self, mat=np.eye(4)):
        # NOTE M*V*P order since all the matrices are in column-major mode.
        self.bufs['prog']["Mvp"].write(
            (mat @ self.vs.view @ self.vs.proj).astype("f4").tobytes())

        self.bufs['fbo'].clear(R, G, B)
        self.bufs['vao'].render()

        img = Image.frombytes('RGB', self.bufs['fbo'].size,
                              self.bufs['fbo'].read(), 'raw', 'RGB', 0, -1)
        return img
