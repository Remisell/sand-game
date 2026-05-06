import taichi as ti
import datetime
import os
import numpy as np
from PIL import Image


@ti.data_oriented
class SandGame:
    def __init__(self, screen_width=1280, screen_height=840, cell_size=1):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = cell_size
        self.grid_width = screen_width // cell_size
        self.grid_height = screen_height // cell_size

        # Инициализируем Taichi
        try:
            ti.init(arch=ti.gpu)
        except:
            print("GPU not available, falling back to CPU")
            ti.init(arch=ti.cpu)

        self.EMPTY = 0
        self.SAND = 1

        # Создаем окно
        self.window = ti.ui.Window("Sand Game", (self.screen_width, self.screen_height),
                                   vsync=True, show_window=True, fps_limit=60)
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()

        # Создаем поля
        self.create_fields()

        # Состояние игры
        self.brush_radius = 5
        self.simulation_speed = 3
        self.sand_color = ti.Vector([0.76, 0.7, 0.5])
        self.bg_color = ti.Vector([0.0, 0.0, 0.0])
        self.show_settings = False
        self.paused = False
        self.clumping_state = False
        self.clumping_strength = 0.5

        # Состояние клавиш
        self.previous_c_state = False
        self.previous_s_state = False
        self.previous_r_state = False
        self.previous_space_state = False

        # Для отслеживания изменения размера
        self.last_width = self.screen_width
        self.last_height = self.screen_height
        self.resize_pending = False

        # Храним ссылки на текущие kernels
        self.update_kernel = None
        self.add_sand_kernel = None
        self.remove_sand_kernel = None
        self.render_kernel = None
        self.reset_kernel = None
        self.draw_brush_kernel = None
        self.clear_brush_kernel = None

        self.undo_stack = []  # Стек для хранения состояний
        self.redo_stack = []  # Стек для повтора отмененных действий
        self.max_undo_states = 50  # Максимальное количество сохраняемых состояний
        self.undo_cooldown = False  # Для предотвращения множественных откатов за один кадр
        self.last_undo_time = 0  # Временная метка последнего отката
        self.previous_z_state = False  # Для отслеживания нажатия Z
        self.previous_ctrl_state = False  # Для отслеживания нажатия Ctrl

        # Инициализация
        self.init()
        self.create_kernels()

    def create_fields(self):
        if self.grid_width <= 0 or self.grid_height <= 0:
            self.grid_width = max(1, self.grid_width)
            self.grid_height = max(1, self.grid_height)

        self.grid = ti.field(dtype=ti.i32, shape=(self.grid_width, self.grid_height))
        self.grid_old = ti.field(dtype=ti.i32, shape=(self.grid_width, self.grid_height))
        self.sand_colors = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_width, self.grid_height))
        self.image_field = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_width, self.grid_height))
        self.brush_overlay = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_width, self.grid_height))

        # Скалярные поля
        self.current_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.background_color = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.clumping_enabled = ti.field(dtype=ti.i32, shape=())
        self.clump_strength = ti.field(dtype=ti.f32, shape=())

    def create_kernels(self):
        @ti.kernel
        def clear_brush():
            for i, j in self.brush_overlay:
                self.brush_overlay[i, j] = ti.Vector([0.0, 0.0, 0.0])

        self.clear_brush_kernel = clear_brush

        @ti.kernel
        def update():
            # Сначала копируем сетку
            for i, j in self.grid:
                self.grid_old[i, j] = self.grid[i, j]
                self.grid[i, j] = self.EMPTY

            # Обновляем позиции песка
            for i, j in ti.ndrange(self.grid_width, self.grid_height):
                if self.grid_old[i, j] == self.SAND:
                    color = self.sand_colors[i, j]
                    moved = False

                    if self.clumping_enabled[None]:
                        has_neighbor_above = j < self.grid_height - 1 and self.grid_old[i, j + 1] == self.SAND
                        has_neighbor_below = j > 0 and self.grid_old[i, j - 1] == self.SAND

                        if (has_neighbor_above or has_neighbor_below) and ti.random() < self.clump_strength[None]:
                            self.grid[i, j] = self.SAND
                            self.sand_colors[i, j] = color
                            continue

                    # Пытаемся переместить вниз
                    if j > 0 and self.grid_old[i, j - 1] == self.EMPTY:
                        self.grid[i, j - 1] = self.SAND
                        self.sand_colors[i, j - 1] = color
                        moved = True
                    else:
                        dir = 1 if ti.random() > 0.5 else -1

                        if (j > 0 and i + dir >= 0 and i + dir < self.grid_width and
                                self.grid_old[i + dir, j - 1] == self.EMPTY):
                            self.grid[i + dir, j - 1] = self.SAND
                            self.sand_colors[i + dir, j - 1] = color
                            moved = True
                        elif (j > 0 and i - dir >= 0 and i - dir < self.grid_width and
                              self.grid_old[i - dir, j - 1] == self.EMPTY):
                            self.grid[i - dir, j - 1] = self.SAND
                            self.sand_colors[i - dir, j - 1] = color
                            moved = True

                    if not moved:
                        self.grid[i, j] = self.SAND
                        self.sand_colors[i, j] = color

        self.update_kernel = update

        @ti.kernel
        def add_sand(x: ti.i32, y: ti.i32, radius: ti.i32):
            color = self.current_color[None]
            radius_sq = radius * radius
            x_start = max(0, x - radius)
            x_end = min(self.grid_width, x + radius + 1)
            y_start = max(0, y - radius)
            y_end = min(self.grid_height, y + radius + 1)

            for i, j in ti.ndrange((x_start, x_end), (y_start, y_end)):
                if (i - x) ** 2 + (j - y) ** 2 < radius_sq:
                    self.grid[i, j] = self.SAND
                    self.sand_colors[i, j] = color

        self.add_sand_kernel = add_sand

        @ti.kernel
        def remove_sand(x: ti.i32, y: ti.i32, radius: ti.i32):
            radius_sq = radius * radius
            x_start = max(0, x - radius)
            x_end = min(self.grid_width, x + radius + 1)
            y_start = max(0, y - radius)
            y_end = min(self.grid_height, y + radius + 1)

            for i, j in ti.ndrange((x_start, x_end), (y_start, y_end)):
                if (i - x) ** 2 + (j - y) ** 2 < radius_sq:
                    if self.grid[i, j] == self.SAND:
                        self.grid[i, j] = self.EMPTY
                        self.sand_colors[i, j] = ti.Vector([0.0, 0.0, 0.0])

        self.remove_sand_kernel = remove_sand

        @ti.kernel
        def render():
            bg_color = self.background_color[None]
            for x, y in ti.ndrange(self.grid_width, self.grid_height):
                material = self.grid[x, y]
                if material == self.EMPTY:
                    self.image_field[x, y] = bg_color + self.brush_overlay[x, y]
                elif material == self.SAND:
                    color = self.sand_colors[x, y]
                    overlay = self.brush_overlay[x, y]
                    if overlay[0] > 0 or overlay[1] > 0 or overlay[2] > 0:
                        self.image_field[x, y] = color * 0.7 + overlay * 0.3
                    else:
                        self.image_field[x, y] = color

        self.render_kernel = render

        @ti.kernel
        def reset():
            for i, j in self.grid:
                self.grid[i, j] = self.EMPTY
                self.sand_colors[i, j] = ti.Vector([0.0, 0.0, 0.0])

        self.reset_kernel = reset

        @ti.kernel
        def draw_brush(x: ti.i32, y: ti.i32, radius: ti.i32, color: ti.types.vector(3, ti.f32)):
            # Очищаем область вокруг кисти
            clear_start_x = max(0, x - radius - 5)
            clear_end_x = min(self.grid_width, x + radius + 6)
            clear_start_y = max(0, y - radius - 5)
            clear_end_y = min(self.grid_height, y + radius + 6)

            for i, j in ti.ndrange((clear_start_x, clear_end_x), (clear_start_y, clear_end_y)):
                self.brush_overlay[i, j] = ti.Vector([0.0, 0.0, 0.0])

            # Рисуем окружность кисти
            radius_f = radius * 1.0
            draw_start_x = max(0, x - radius - 2)
            draw_end_x = min(self.grid_width, x + radius + 3)
            draw_start_y = max(0, y - radius - 2)
            draw_end_y = min(self.grid_height, y + radius + 3)

            for i, j in ti.ndrange((draw_start_x, draw_end_x), (draw_start_y, draw_end_y)):
                dist = ti.sqrt((i - x) ** 2 + (j - y) ** 2)
                # Рисуем окружность
                if ti.abs(dist - radius_f) < 1.5:
                    self.brush_overlay[i, j] = color
                # Рисуем крестик в центре
                if (i == x or j == y) and ti.abs(i - x) <= 2 and ti.abs(j - y) <= 2:
                    self.brush_overlay[i, j] = color

        self.draw_brush_kernel = draw_brush

    @ti.kernel
    def init(self):
        for i, j in self.grid:
            self.grid[i, j] = self.EMPTY
            self.sand_colors[i, j] = ti.Vector([0.0, 0.0, 0.0])
            self.brush_overlay[i, j] = ti.Vector([0.0, 0.0, 0.0])
        self.current_color[None] = ti.Vector([0.76, 0.7, 0.5])
        self.background_color[None] = ti.Vector([0.0, 0.0, 0.0])
        self.clumping_enabled[None] = 0
        self.clump_strength[None] = 0.5

    def recreate_with_new_size(self, new_width, new_height):
        # Сохраняем старые данные
        old_grid_data = None
        old_colors_data = None

        if hasattr(self, 'grid'):
            try:
                old_grid_data = self.grid.to_numpy()
                old_colors_data = self.sand_colors.to_numpy()
            except:
                pass

        # Обновляем размеры
        old_grid_width = self.grid_width
        old_grid_height = self.grid_height
        self.screen_width = new_width
        self.screen_height = new_height
        self.grid_width = max(1, new_width // self.cell_size)
        self.grid_height = max(1, new_height // self.cell_size)

        # Создаем новые поля
        self.create_fields()

        # Копируем старые данные если они есть
        if old_grid_data is not None and old_colors_data is not None:
            copy_width = min(old_grid_width, self.grid_width)
            copy_height = min(old_grid_height, self.grid_height)

            # Используем numpy для быстрого копирования
            new_grid = np.zeros((self.grid_width, self.grid_height), dtype=np.int32)
            new_colors = np.zeros((self.grid_width, self.grid_height, 3), dtype=np.float32)

            new_grid[:copy_width, :copy_height] = old_grid_data[:copy_width, :copy_height]
            new_colors[:copy_width, :copy_height] = old_colors_data[:copy_width, :copy_height]

            self.grid.from_numpy(new_grid)
            self.sand_colors.from_numpy(new_colors)

        # Сбрасываем overlay
        self.clear_brush()

        # Обновляем скалярные поля
        self.current_color[None] = self.sand_color
        self.background_color[None] = self.bg_color
        self.clumping_enabled[None] = 1 if self.clumping_state else 0
        self.clump_strength[None] = self.clumping_strength

        # Пересоздаем kernels с новыми размерами
        self.create_kernels()

        # Очищаем стеки отката при изменении размера
        self.undo_stack.clear()
        self.redo_stack.clear()

        # Сохраняем начальное состояние
        self.save_state()

    def clear_brush(self):
        if self.clear_brush_kernel:
            self.clear_brush_kernel()
            ti.sync()

    def update(self):
        if self.update_kernel:
            self.update_kernel()
            ti.sync()

    def add_sand(self, x, y, radius):
        if self.add_sand_kernel:
            self.add_sand_kernel(x, y, radius)
            ti.sync()

    def remove_sand(self, x, y, radius):
        if self.remove_sand_kernel:
            self.remove_sand_kernel(x, y, radius)
            ti.sync()

    def render(self):
        if self.render_kernel:
            self.render_kernel()
            ti.sync()

    def reset(self):
        # Сохраняем состояние перед сбросом
        self.auto_save_state("reset")
        if self.reset_kernel:
            self.reset_kernel()
            ti.sync()

    def draw_brush(self, x, y, radius, color):
        if self.draw_brush_kernel:
            self.draw_brush_kernel(x, y, radius, color)
            ti.sync()

    def save_screenshot(self):
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshots_dir}/sand_game_{timestamp}.png"

        self.render()  # Убеждаемся что image_field обновлен
        img_data = self.image_field.to_numpy()
        img_data = np.transpose(img_data, (1, 0, 2))
        img_data = np.flipud(img_data)
        img_data = (img_data * 255).astype(np.uint8)

        img = Image.fromarray(img_data, 'RGB')
        img.save(filename, 'PNG')
        print(f"Скриншот сохранен: {filename}")

        return filename

    def save_state(self):
        if len(self.undo_stack) >= self.max_undo_states:
            self.undo_stack.pop(0)  # Удаляем самое старое состояние

        # Сохраняем состояние сетки и цветов
        state = {
            'grid': self.grid.to_numpy().copy(),
            'colors': self.sand_colors.to_numpy().copy()
        }
        self.undo_stack.append(state)
        # Очищаем стек повтора при новом действии
        self.redo_stack.clear()

    def load_state(self, state):
        if state:
            self.grid.from_numpy(state['grid'])
            self.sand_colors.from_numpy(state['colors'])

    def undo(self):
        if len(self.undo_stack) > 0:
            # Сохраняем текущее состояние в стек повтора
            current_state = {
                'grid': self.grid.to_numpy().copy(),
                'colors': self.sand_colors.to_numpy().copy()
            }
            self.redo_stack.append(current_state)

            # Загружаем последнее сохраненное состояние
            last_state = self.undo_stack.pop()
            self.load_state(last_state)

            # Очищаем кисть после отката
            self.clear_brush()

    def redo(self):
        if len(self.redo_stack) > 0:
            # Сохраняем текущее состояние в стек отката
            current_state = {
                'grid': self.grid.to_numpy().copy(),
                'colors': self.sand_colors.to_numpy().copy()
            }
            self.undo_stack.append(current_state)

            # Загружаем состояние из стека повтора
            next_state = self.redo_stack.pop()
            self.load_state(next_state)

            # Очищаем кисть после повтора
            self.clear_brush()

    def auto_save_state(self):
        # Проверяем, не слишком ли часто сохраняем
        import time
        current_time = time.time()
        if current_time - self.last_undo_time > 0.1:  # Не чаще чем раз в 100ms
            self.save_state()
            self.last_undo_time = current_time
            return True
        return False

    def clear_history(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        # Сохраняем текущее состояние как начальное
        self.save_state()

    def settings(self):
        self.gui.begin("Settings", 0.05, 0.05, 0.3, 0.85)

        self.gui.text("=== Sand Color ===")
        r = self.sand_color[0]
        g = self.sand_color[1]
        b = self.sand_color[2]

        new_r = self.gui.slider_float("##sand_red", r, 0.0, 1.0)
        new_g = self.gui.slider_float("##sand_green", g, 0.0, 1.0)
        new_b = self.gui.slider_float("##sand_blue", b, 0.0, 1.0)

        self.gui.text("=== Background Color ===")
        bg_r = self.bg_color[0]
        bg_g = self.bg_color[1]
        bg_b = self.bg_color[2]

        new_bg_r = self.gui.slider_float("##bg_red", bg_r, 0.0, 1.0)
        new_bg_g = self.gui.slider_float("##bg_green", bg_g, 0.0, 1.0)
        new_bg_b = self.gui.slider_float("##bg_blue", bg_b, 0.0, 1.0)

        self.gui.text("=== Brush ===")
        new_brush_radius = self.gui.slider_int("##brush", self.brush_radius, 1, 40)

        self.gui.text("=== Simulation ===")
        new_speed = self.gui.slider_int("##speed", self.simulation_speed, 0, 10)

        self.gui.text("=== Sand Clumping ===")
        new_clumping_state = self.gui.checkbox("Enable Clumping", self.clumping_state)
        if new_clumping_state:
            self.gui.text("Clumping Strength:")
            new_clumping_strength = self.gui.slider_float("##clump_strength", self.clumping_strength, 0.0, 1.0)
        else:
            new_clumping_strength = self.clumping_strength

        self.gui.end()

        # Обновляем значения
        self.sand_color = ti.Vector([new_r, new_g, new_b])
        self.bg_color = ti.Vector([new_bg_r, new_bg_g, new_bg_b])
        self.brush_radius = new_brush_radius
        self.simulation_speed = new_speed
        self.clumping_state = new_clumping_state
        self.clumping_strength = new_clumping_strength

        # Применяем к полям Taichi
        self.current_color[None] = self.sand_color
        self.background_color[None] = self.bg_color
        self.clumping_enabled[None] = 1 if self.clumping_state else 0
        self.clump_strength[None] = self.clumping_strength

    def welcome(self):
        self.gui.begin("Welcome to Sand Game!", 0.25, 0.25, 0.5, 0.5)
        self.gui.text("=== Controls ===")
        self.gui.text("LMB: Add sand")
        self.gui.text("RMB: Remove sand")
        self.gui.text("SPACE: Pause/Unpause")
        self.gui.text("C: Toggle settings")
        self.gui.text("S: Take screenshot")
        self.gui.text("R: Reset")
        self.gui.text("Ctrl+Z: Undo")
        self.gui.text("Ctrl+S: Save current state")
        self.gui.text("Ctrl+Y or Ctrl+Shift+Z: Redo")
        self.gui.text("ESC: Quit")
        self.gui.text("")
        self.gui.text("=== Tips ===")
        self.gui.text("- Adjust brush size in settings")
        self.gui.text("- Change sand color in settings")
        self.gui.text("- Enable clumping for sticky sand")
        self.gui.text("")

        button_pressed = self.gui.button("Start Game")
        self.gui.end()
        return button_pressed

    def run(self):
        welcome_active = True
        frame_count = 0
        last_mouse_x, last_mouse_y = -1, -1

        while self.window.running:
            frame_count += 1

            # Проверяем изменение размера окна
            current_width, current_height = self.window.get_window_shape()
            if (current_width != self.last_width or current_height != self.last_height):
                self.recreate_with_new_size(current_width, current_height)
                self.last_width = current_width
                self.last_height = current_height
                last_mouse_x, last_mouse_y = -1, -1
                continue

            if self.window.is_pressed(ti.ui.ESCAPE):
                break

            # Получаем позицию мыши
            mouse_pos = self.window.get_cursor_pos()
            grid_x = int(mouse_pos[0] * self.grid_width)
            grid_y = int(mouse_pos[1] * self.grid_height)

            # Проверка границ
            grid_x = max(0, min(grid_x, self.grid_width - 1))
            grid_y = max(0, min(grid_y, self.grid_height - 1))

            if not welcome_active:
                # Проверка нажатия Ctrl
                ctrl_pressed = (self.window.is_pressed(ti.ui.CTRL) or
                                self.window.is_pressed('ctrl') or
                                self.window.is_pressed('control'))

                # Проверка нажатия Shift
                shift_pressed = (self.window.is_pressed(ti.ui.SHIFT) or
                                 self.window.is_pressed('shift'))

                # Обработка клавиш
                current_c_state = self.window.is_pressed('c') or self.window.is_pressed('C')
                if current_c_state and not self.previous_c_state:
                    self.show_settings = not self.show_settings
                self.previous_c_state = current_c_state

                current_s_state = self.window.is_pressed('s') or self.window.is_pressed('S')
                if current_s_state and not self.previous_s_state:
                    if ctrl_pressed:
                        # Ctrl+S - сохраняем состояние вручную
                        self.save_state()
                        print("Состояние сохранено (Ctrl+S)")
                    else:
                        self.save_screenshot()
                self.previous_s_state = current_s_state

                current_r_state = self.window.is_pressed('r') or self.window.is_pressed('R')
                if current_r_state and not self.previous_r_state:
                    if not ctrl_pressed:  # Обычный R - сброс
                        # Сохраняем состояние перед сбросом
                        self.auto_save_state()
                        self.reset()
                        self.clear_brush()
                self.previous_r_state = current_r_state

                # Обработка Ctrl+Z (Undo)
                current_z_state = self.window.is_pressed('z') or self.window.is_pressed('Z')
                if current_z_state and not self.previous_z_state and ctrl_pressed:
                    if shift_pressed or ctrl_pressed:  # Ctrl+Shift+Z или Ctrl+Y для Redo
                        if shift_pressed:  # Ctrl+Shift+Z
                            self.redo()
                        else:  # Просто Ctrl+Z
                            self.undo()
                    self.clear_brush()  # Очищаем кисть после отмены/повтора
                self.previous_z_state = current_z_state

                # Обработка Ctrl+Y (альтернатива для Redo)
                current_y_state = self.window.is_pressed('y') or self.window.is_pressed('Y')
                if current_y_state and not hasattr(self, 'previous_y_state'):
                    self.previous_y_state = False
                if current_y_state and not self.previous_y_state and ctrl_pressed:
                    self.redo()
                    self.clear_brush()
                self.previous_y_state = current_y_state

                current_space_state = self.window.is_pressed(ti.ui.SPACE)
                if current_space_state and not self.previous_space_state:
                    self.paused = not self.paused
                self.previous_space_state = current_space_state

                # Рисуем кисть только если мышь переместилась
                if grid_x != last_mouse_x or grid_y != last_mouse_y:
                    self.clear_brush()
                    brush_color = ti.Vector([1.0, 1.0, 1.0])
                    self.draw_brush(grid_x, grid_y, self.brush_radius, brush_color)
                    last_mouse_x, last_mouse_y = grid_x, grid_y

                if self.show_settings:
                    self.settings()
                else:
                    # Рисование песка
                    if self.window.is_pressed(ti.ui.LMB):
                        # Сохраняем состояние перед добавлением песка (но не слишком часто)
                        if not hasattr(self, '_last_save_frame') or frame_count - self._last_save_frame > 30:
                            self.auto_save_state()
                            self._last_save_frame = frame_count
                        self.add_sand(grid_x, grid_y, self.brush_radius)

                    if self.window.is_pressed(ti.ui.RMB):
                        # Сохраняем состояние перед удалением песка
                        if not hasattr(self, '_last_save_frame') or frame_count - self._last_save_frame > 30:
                            self.auto_save_state()
                            self._last_save_frame = frame_count
                        self.remove_sand(grid_x, grid_y, self.brush_radius)

                # Обновление симуляции
                if not self.paused:
                    for _ in range(self.simulation_speed):
                        self.update()
            else:
                welcome_active = not self.welcome()
                if not welcome_active:
                    self.clear_brush()
                    # Сохраняем начальное состояние при старте
                    self.save_state()

            # Рендеринг
            self.render()
            self.canvas.set_image(self.image_field)

            self.window.show()


def get_optimal_window_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
    except:
        try:
            import pyautogui
            screen_width, screen_height = pyautogui.size()
        except:
            screen_width, screen_height = 1280, 840

    window_width = min(screen_width - 100, 1920)
    window_height = min(screen_height - 100, 1080)
    window_width = max(window_width, 800)
    window_height = max(window_height, 600)

    return window_width, window_height


def main():
    window_width, window_height = get_optimal_window_size()
    game = SandGame(window_width, window_height, cell_size=1)
    game.run()


if __name__ == "__main__":
    main()