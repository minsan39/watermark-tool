import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, Frame, Label
from tkinter import ttk
from PIL import Image, ImageTk
from simple_lama_inpainting import SimpleLama
import torch
import threading

class WatermarkRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Remover")
        
        self.image = None
        self.original_image = None
        self.display_image = None
        self.tk_image = None
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.roi_selected = False
        self.base_scale = 1.0
        self.zoom_scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.canvas_w = 800
        self.canvas_h = 500
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.simple_lama = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)
        
        Button(btn_frame, text="Open", command=self._open_image, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="Remove", command=self._remove_watermark, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="Save", command=self._save_image, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="Reset", command=self._reset_image, width=10).pack(side="left", padx=5)
        
        self.canvas = Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="gray")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="Click [Open] to load an image", fill="white", font=("Arial", 14))
        
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        
        self.status_label = Frame(self.root, bd=1, relief="sunken")
        self.status_label.pack(fill="x", padx=10, pady=5)
        self.status_text = Label(self.status_label, text="Ready | Scroll to zoom, Right-drag to pan", anchor="w")
        self.status_text.pack(fill="x", padx=5)
        
        self.progress_frame = Frame(self.root)
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("green.Horizontal.TProgressbar", troughcolor='#d0d0d0', background='#00c853')
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient="horizontal", length=100, mode="indeterminate", style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(fill="x")
        self.progress_bar.pack_forget()
        
    def _open_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                self.image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if self.image is not None:
                    self.original_image = self.image.copy()
                    self.roi_selected = False
                    self.zoom_scale = 1.0
                    self._update_display("Image loaded. Drag to select watermark area.")
                else:
                    self._update_status("Failed to decode image!")
            except Exception as e:
                self._update_status(f"Error: {str(e)}")
                
    def _remove_watermark(self):
        if self.image is None:
            self._update_status("Please open an image first!")
            return
        if not self.roi_selected:
            self._update_status("Please select watermark area first!")
            return
        
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("Processing with LaMa AI model...")
        self.root.update()
        
        threading.Thread(target=self._process_in_background, daemon=True).start()
    
    def _process_in_background(self):
        try:
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("Loading LaMa model (first time will download ~200MB)..."))
                self.simple_lama = SimpleLama()
                self.root.after(0, lambda: self._update_status("Processing with LaMa AI model..."))
                
            real_ix = int((self.ix - self.offset_x) / (self.base_scale * self.zoom_scale))
            real_iy = int((self.iy - self.offset_y) / (self.base_scale * self.zoom_scale))
            real_fx = int((self.fx - self.offset_x) / (self.base_scale * self.zoom_scale))
            real_fy = int((self.fy - self.offset_y) / (self.base_scale * self.zoom_scale))
            
            x1, x2 = min(real_ix, real_fx), max(real_ix, real_fx)
            y1, y2 = min(real_iy, real_fy), max(real_iy, real_fy)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.image.shape[1], x2)
            y2 = min(self.image.shape[0], y2)
            
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            mask_pil = Image.fromarray(mask).convert("L")
            
            result_pil = self.simple_lama(image_pil, mask_pil)
            
            result_array = np.array(result_pil)
            new_image = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            self.root.after(0, lambda: self._finish_processing(new_image))
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _finish_processing(self, new_image):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.image = new_image
        self.roi_selected = False
        self._refresh_display("Watermark removed! Click [Save] to save.")
    
    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._update_status(f"Error: {error_msg}")
    
    def _inpaint_telea_adaptive(self, image, mask):
        result = image.copy()
        mask_binary = (mask > 0).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return result
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            expand = 5
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(image.shape[1], x + w + expand)
            y2 = min(image.shape[0], y + h + expand)
            
            local_mask = mask_binary[y1:y2, x1:x2]
            local_image = result[y1:y2, x1:x2]
            
            local_result = cv2.inpaint(local_image, local_mask, 3, cv2.INPAINT_TELEA)
            result[y1:y2, x1:x2] = local_result
        
        return result
        
    def _save_image(self):
        if self.image is None:
            self._update_status("No image to save!")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
        )
        if filepath:
            cv2.imwrite(filepath, self.image)
            self._update_status(f"Saved: {filepath}")
            
    def _reset_image(self):
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.roi_selected = False
            self.zoom_scale = 1.0
            self._update_display("Image reset. Drag to select watermark area.")
            
    def _update_display(self, message=""):
        if self.image is None:
            return
            
        h, w = self.image.shape[:2]
        self.base_scale = min(self.canvas_w / w, self.canvas_h / h, 1.0)
        
        total_scale = self.base_scale * self.zoom_scale
        new_w = int(w * total_scale)
        new_h = int(h * total_scale)
        
        self.offset_x = (self.canvas_w - new_w) // 2
        self.offset_y = (self.canvas_h - new_h) // 2
        
        self._refresh_display(message)
        
    def _refresh_display(self, message=""):
        if self.image is None:
            return
            
        h, w = self.image.shape[:2]
        total_scale = self.base_scale * self.zoom_scale
        new_w = int(w * total_scale)
        new_h = int(h * total_scale)
        
        display = cv2.resize(self.image.copy(), (new_w, new_h))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        self.display_image = Image.fromarray(display)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        
        zoom_pct = int(self.zoom_scale * 100)
        if message:
            self._update_status(f"{message} | Zoom: {zoom_pct}%")
        else:
            self._update_status(f"Zoom: {zoom_pct}%")
            
    def _update_status(self, message):
        self.status_text.config(text=message)
            
    def _on_mouse_down(self, event):
        if self.image is None:
            return
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        self.fx, self.fy = event.x, event.y
        
    def _on_mouse_move(self, event):
        if not self.drawing or self.image is None:
            return
        self.fx, self.fy = event.x, event.y
        self._draw_selection()
        
    def _on_mouse_up(self, event):
        if not self.drawing or self.image is None:
            return
        self.drawing = False
        self.fx, self.fy = event.x, event.y
        self.roi_selected = True
        self._draw_selection()
        self._update_status("Area selected. Click [Remove] to remove watermark.")
        
    def _draw_selection(self):
        self.canvas.delete("selection")
        self.canvas.create_rectangle(
            self.ix, self.iy, self.fx, self.fy,
            outline="green", width=2, tags="selection"
        )
        
    def _on_pan_start(self, event):
        if self.image is None:
            return
        self.panning = True
        self.pan_start_x = event.x - self.offset_x
        self.pan_start_y = event.y - self.offset_y
        
    def _on_pan_move(self, event):
        if not self.panning or self.image is None:
            return
        self.offset_x = event.x - self.pan_start_x
        self.offset_y = event.y - self.pan_start_y
        self._redraw_image()
        
    def _on_pan_end(self, event):
        self.panning = False
        
    def _redraw_image(self):
        if self.image is None or self.display_image is None:
            return
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.tk_image)
        
    def _on_mouse_wheel(self, event):
        if self.image is None:
            return
            
        old_scale = self.zoom_scale
        
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.zoom_scale *= 1.2
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.zoom_scale /= 1.2
            
        self.zoom_scale = max(0.1, min(self.zoom_scale, 10.0))
        
        mouse_x = event.x
        mouse_y = event.y
        
        img_x = (mouse_x - self.offset_x) / old_scale
        img_y = (mouse_y - self.offset_y) / old_scale
        
        self.offset_x = mouse_x - img_x * self.zoom_scale
        self.offset_y = mouse_y - img_y * self.zoom_scale
        
        h, w = self.image.shape[:2]
        total_scale = self.base_scale * self.zoom_scale
        new_w = int(w * total_scale)
        new_h = int(h * total_scale)
        
        display = cv2.resize(self.image.copy(), (new_w, new_h))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        
        self.display_image = Image.fromarray(display)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        self._redraw_image()
        zoom_pct = int(self.zoom_scale * 100)
        self._update_status(f"Zoom: {zoom_pct}%")

if __name__ == "__main__":
    root = Tk()
    app = WatermarkRemover(root)
    root.mainloop()
