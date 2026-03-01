import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Canvas, Frame, Label, Entry, StringVar, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk
from simple_lama_inpainting import SimpleLama
import threading
from rapidocr_onnxruntime import RapidOCR

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
        self.ocr_reader = None
        self.mode = "box"
        self.template_image = None
        self.template_path = None
        
        self.image_list = []
        self.original_image_list = []
        self.image_paths = []
        self.current_image_index = -1
        self.batch_processing = False
        
        self._setup_ui()
        
    def _setup_ui(self):
        btn_frame = Frame(self.root)
        btn_frame.pack(pady=10)
        
        Button(btn_frame, text="Open", command=self._open_image, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="Remove", command=self._remove_watermark, width=10).pack(side="left", padx=5)
        self.btn_batch_remove = Button(btn_frame, text="Batch Remove", command=self._batch_remove_watermark, width=12, state="disabled")
        self.btn_batch_remove.pack(side="left", padx=5)
        Button(btn_frame, text="Save", command=self._save_image, width=10).pack(side="left", padx=5)
        self.btn_batch_save = Button(btn_frame, text="Batch Save", command=self._batch_save_image, width=10, state="disabled")
        self.btn_batch_save.pack(side="left", padx=5)
        Button(btn_frame, text="Reset", command=self._reset_image, width=10).pack(side="left", padx=5)
        
        nav_frame = Frame(self.root)
        nav_frame.pack(pady=5)
        
        self.btn_prev = Button(nav_frame, text="◀ Prev", command=self._prev_image, width=10, state="disabled")
        self.btn_prev.pack(side="left", padx=5)
        
        self.image_counter_label = Label(nav_frame, text="0 / 0", width=10)
        self.image_counter_label.pack(side="left", padx=5)
        
        self.btn_next = Button(nav_frame, text="Next ▶", command=self._next_image, width=10, state="disabled")
        self.btn_next.pack(side="left", padx=5)
        
        mode_frame = Frame(self.root)
        mode_frame.pack(pady=5)
        
        self.mode_var = StringVar(value="box")
        Label(mode_frame, text="Mode:").pack(side="left", padx=5)
        
        ttk.Radiobutton(mode_frame, text="Box Select", variable=self.mode_var, 
                        value="box", command=self._switch_mode).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Text", variable=self.mode_var, 
                        value="text", command=self._switch_mode).pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Image", variable=self.mode_var, 
                        value="image", command=self._switch_mode).pack(side="left", padx=5)
        
        self.text_frame = Frame(self.root)
        Label(self.text_frame, text="Watermark Text:").pack(side="left", padx=5)
        self.text_entry = Entry(self.text_frame, width=40)
        self.text_entry.pack(side="left", padx=5)
        self.text_frame.pack(pady=5)
        self.text_frame.pack_forget()
        
        self.image_frame = Frame(self.root)
        Label(self.image_frame, text="Template:").pack(side="left", padx=5)
        Button(self.image_frame, text="Load Template", command=self._load_template, width=15).pack(side="left", padx=5)
        self.template_label = Label(self.image_frame, text="No template loaded", fg="gray")
        self.template_label.pack(side="left", padx=5)
        
        self.threshold_frame = Frame(self.root)
        Label(self.threshold_frame, text="Threshold:").pack(side="left", padx=5)
        self.threshold_var = StringVar(value="0.3")
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0.1, to=0.9, variable=self.threshold_var, orient="horizontal", length=150)
        self.threshold_scale.pack(side="left", padx=5)
        self.threshold_label = Label(self.threshold_frame, text="0.30", width=5)
        self.threshold_label.pack(side="left", padx=5)
        self.threshold_scale.bind("<Motion>", self._update_threshold_label)
        self.threshold_scale.bind("<ButtonRelease-1>", self._update_threshold_label)
        self.threshold_frame.pack(pady=5)
        self.threshold_frame.pack_forget()
        self.image_frame.pack(pady=5)
        self.image_frame.pack_forget()
        
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
        
    def _update_threshold_label(self, event=None):
        val = float(self.threshold_var.get())
        self.threshold_label.config(text=f"{val:.2f}")
    
    def _update_nav_buttons(self):
        total = len(self.image_list)
        if total > 0:
            self.image_counter_label.config(text=f"{self.current_image_index + 1} / {total}")
            self.btn_prev.config(state="normal" if self.current_image_index > 0 else "disabled")
            self.btn_next.config(state="normal" if self.current_image_index < total - 1 else "disabled")
            self.btn_batch_remove.config(state="normal" if total > 1 else "disabled")
            self.btn_batch_save.config(state="normal" if total > 1 else "disabled")
        else:
            self.image_counter_label.config(text="0 / 0")
            self.btn_prev.config(state="disabled")
            self.btn_next.config(state="disabled")
            self.btn_batch_remove.config(state="disabled")
            self.btn_batch_save.config(state="disabled")
    
    def _update_batch_display(self, current_idx, total):
        self._update_nav_buttons()
        self._refresh_display(f"Processing {current_idx + 1}/{total}...")
    
    def _prev_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.image = self.image_list[self.current_image_index]
            self.original_image = self.original_image_list[self.current_image_index]
            self.roi_selected = False
            self.zoom_scale = 1.0
            self._update_nav_buttons()
            self._update_display(f"Image {self.current_image_index + 1}/{len(self.image_list)}")
    
    def _next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.image = self.image_list[self.current_image_index]
            self.original_image = self.original_image_list[self.current_image_index]
            self.roi_selected = False
            self.zoom_scale = 1.0
            self._update_nav_buttons()
            self._update_display(f"Image {self.current_image_index + 1}/{len(self.image_list)}")
        
    def _switch_mode(self):
        self.mode = self.mode_var.get()
        self.roi_selected = False
        
        self.text_frame.pack_forget()
        self.image_frame.pack_forget()
        self.threshold_frame.pack_forget()
        
        if self.mode == "text":
            self.text_frame.pack(pady=5)
            self._update_status("Text mode: Enter watermark text and click [Remove]")
        elif self.mode == "image":
            self.image_frame.pack(pady=5)
            self.threshold_frame.pack(pady=5)
            if self.template_image is not None:
                self._update_status("Image mode: Click [Remove] to find and remove watermark")
            else:
                self._update_status("Image mode: Load a template image first")
        else:
            self._update_status("Box mode: Drag to select watermark area")
        
        if self.image is not None:
            self._refresh_display("")
            
    def _load_template(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if filepath:
            try:
                with open(filepath, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                full_template = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if full_template is not None:
                    self._show_template_crop_window(full_template)
                else:
                    self._update_status("Failed to decode template image!")
            except Exception as e:
                self._update_status(f"Error: {str(e)}")
    
    def _show_template_crop_window(self, full_template):
        crop_window = Toplevel(self.root)
        crop_window.title("Select Watermark Region")
        
        h, w = full_template.shape[:2]
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        display_w = int(w * scale)
        display_h = int(h * scale)
        
        crop_window.geometry(f"{display_w + 20}x{display_h + 100}")
        
        Label(crop_window, text="Drag to select the watermark area, then click Confirm").pack(pady=5)
        
        crop_canvas = Canvas(crop_window, width=display_w, height=display_h, bg="gray")
        crop_canvas.pack(padx=10, pady=5)
        
        display_template = cv2.resize(full_template, (display_w, display_h))
        display_template = cv2.cvtColor(display_template, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(display_template)
        tk_img = ImageTk.PhotoImage(pil_img)
        
        crop_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        
        crop_data = {"start": None, "end": None, "drawing": False}
        
        def on_crop_down(event):
            crop_data["drawing"] = True
            crop_data["start"] = (event.x, event.y)
            crop_data["end"] = (event.x, event.y)
        
        def on_crop_move(event):
            if not crop_data["drawing"]:
                return
            crop_data["end"] = (event.x, event.y)
            crop_canvas.delete("crop")
            crop_canvas.create_rectangle(
                crop_data["start"][0], crop_data["start"][1],
                crop_data["end"][0], crop_data["end"][1],
                outline="red", width=2, tags="crop"
            )
        
        def on_crop_up(event):
            crop_data["drawing"] = False
            crop_data["end"] = (event.x, event.y)
        
        crop_canvas.bind("<ButtonPress-1>", on_crop_down)
        crop_canvas.bind("<B1-Motion>", on_crop_move)
        crop_canvas.bind("<ButtonRelease-1>", on_crop_up)
        
        def confirm_crop():
            if crop_data["start"] and crop_data["end"]:
                x1 = int(min(crop_data["start"][0], crop_data["end"][0]) / scale)
                y1 = int(min(crop_data["start"][1], crop_data["end"][1]) / scale)
                x2 = int(max(crop_data["start"][0], crop_data["end"][0]) / scale)
                y2 = int(max(crop_data["start"][1], crop_data["end"][1]) / scale)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    self.template_image = full_template[y1:y2, x1:x2]
                    th, tw = self.template_image.shape[:2]
                    self.template_label.config(text=f"{tw}x{th} cropped", fg="green")
                    self._update_status("Template cropped. Click [Remove] to find and remove watermark.")
                    crop_window.destroy()
                else:
                    self._update_status("Invalid selection!")
            else:
                self._update_status("Please select a region first!")
        
        btn_frame = Frame(crop_window)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="Confirm", command=confirm_crop, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="Cancel", command=crop_window.destroy, width=10).pack(side="left", padx=5)
        
        crop_window.transient(self.root)
        crop_window.grab_set()
        self.root.wait_window(crop_window)
            
    def _open_image(self):
        filepaths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if filepaths:
            existing_count = len(self.image_list)
            failed_files = []
            new_count = 0
            
            for filepath in filepaths:
                try:
                    with open(filepath, 'rb') as f:
                        img_array = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        self.image_list.append(img)
                        self.original_image_list.append(img.copy())
                        self.image_paths.append(filepath)
                        new_count += 1
                    else:
                        failed_files.append(filepath)
                except Exception as e:
                    failed_files.append(filepath)
            
            if new_count > 0:
                self.current_image_index = len(self.image_list) - 1
                self.image = self.image_list[self.current_image_index]
                self.original_image = self.original_image_list[self.current_image_index]
                self.roi_selected = False
                self.zoom_scale = 1.0
                
                total_count = len(self.image_list)
                count_msg = f"Added {new_count} image(s), total: {total_count}"
                if failed_files:
                    count_msg += f", {len(failed_files)} failed"
                
                self._update_nav_buttons()
                msg = f"{count_msg}. "
                if self.mode == "text":
                    msg += "Enter watermark text."
                elif self.mode == "image":
                    msg += "Load template and click Remove." if self.template_image is None else "Click Remove to find watermark."
                else:
                    msg += "Drag to select watermark area."
                self._update_display(msg)
            else:
                self._update_status("No valid images loaded!")
                
    def _remove_watermark(self):
        if self.image is None:
            self._update_status("Please open an image first!")
            return
        
        if self.mode == "box":
            if not self.roi_selected:
                self._update_status("Please select watermark area first!")
                return
            self._start_box_removal()
        elif self.mode == "text":
            watermark_text = self.text_entry.get().strip()
            if not watermark_text:
                self._update_status("Please enter watermark text!")
                return
            self._start_text_removal(watermark_text)
        else:
            if self.template_image is None:
                self._update_status("Please load a template image first!")
                return
            self._start_image_removal()
    
    def _batch_remove_watermark(self):
        if self.image is None:
            self._update_status("Please open images first!")
            return
        
        if len(self.image_list) <= 1:
            self._update_status("Batch Remove requires multiple images!")
            return
        
        if self.mode == "box":
            self._update_status("Box mode does not support batch processing. Use Text or Image mode.")
            return
        elif self.mode == "text":
            watermark_text = self.text_entry.get().strip()
            if not watermark_text:
                self._update_status("Please enter watermark text!")
                return
            self._start_batch_text_removal(watermark_text)
        else:
            if self.template_image is None:
                self._update_status("Please load a template image first!")
                return
            self._start_batch_image_removal()
    
    def _start_box_removal(self):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("Processing with LaMa AI model...")
        self.root.update()
        
        threading.Thread(target=self._process_box_in_background, daemon=True).start()
    
    def _start_text_removal(self, watermark_text):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("Initializing OCR and detecting text...")
        self.root.update()
        
        threading.Thread(target=self._process_text_in_background, args=(watermark_text,), daemon=True).start()
    
    def _start_image_removal(self):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("Searching for watermark pattern...")
        self.root.update()
        
        threading.Thread(target=self._process_image_in_background, daemon=True).start()
    
    def _start_batch_text_removal(self, watermark_text):
        self.batch_processing = True
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status(f"Batch processing: 0/{len(self.image_list)} images...")
        self.root.update()
        
        threading.Thread(target=self._process_batch_text_in_background, args=(watermark_text,), daemon=True).start()
    
    def _start_batch_image_removal(self):
        self.batch_processing = True
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status(f"Batch processing: 0/{len(self.image_list)} images...")
        self.root.update()
        
        threading.Thread(target=self._process_batch_image_in_background, daemon=True).start()
    
    def _process_box_in_background(self):
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
            
            self._apply_lama_inpaint(mask)
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _process_text_in_background(self, watermark_text):
        try:
            if self.ocr_reader is None:
                self.root.after(0, lambda: self._update_status("Loading OCR model..."))
                self.ocr_reader = RapidOCR()
                self.root.after(0, lambda: self._update_status("Detecting text in image..."))
            
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            result, elapse = self.ocr_reader(image_rgb)
            
            if result is None or len(result) == 0:
                self.root.after(0, lambda: self._handle_error("No text found in image"))
                return
            
            matched_boxes = []
            watermark_lower = watermark_text.lower().replace(" ", "").replace("AI", "ai")
            
            for detection in result:
                bbox = detection[0]
                text = detection[1]
                confidence = detection[2] if len(detection) > 2 else 1.0
                
                text_clean = text.lower().replace(" ", "").replace("AI", "ai")
                
                if watermark_lower in text_clean or text_clean in watermark_lower:
                    matched_boxes.append(bbox)
                    self.root.after(0, lambda t=text: self._update_status(f"Found matching text: '{t}'"))
            
            if not matched_boxes:
                self.root.after(0, lambda: self._handle_error(f"Text '{watermark_text}' not found in image"))
                return
            
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            
            for bbox in matched_boxes:
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            self.root.after(0, lambda: self._update_status("Processing with LaMa AI model..."))
            self._apply_lama_inpaint(mask)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _process_batch_text_in_background(self, watermark_text):
        try:
            if self.ocr_reader is None:
                self.root.after(0, lambda: self._update_status("Loading OCR model..."))
                self.ocr_reader = RapidOCR()
            
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("Loading LaMa model..."))
                self.simple_lama = SimpleLama()
            
            total = len(self.image_list)
            success_count = 0
            fail_count = 0
            watermark_lower = watermark_text.lower().replace(" ", "").replace("AI", "ai")
            
            for idx, img in enumerate(self.image_list):
                self.current_image_index = idx
                self.image = self.image_list[idx]
                self.original_image = self.original_image_list[idx]
                self.root.after(0, lambda i=idx: self._update_batch_display(i, total))
                
                try:
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result, elapse = self.ocr_reader(image_rgb)
                    
                    if result is None or len(result) == 0:
                        fail_count += 1
                        continue
                    
                    matched_boxes = []
                    for detection in result:
                        bbox = detection[0]
                        text = detection[1]
                        text_clean = text.lower().replace(" ", "").replace("AI", "ai")
                        
                        if watermark_lower in text_clean or text_clean in watermark_lower:
                            matched_boxes.append(bbox)
                    
                    if not matched_boxes:
                        fail_count += 1
                        continue
                    
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    for bbox in matched_boxes:
                        pts = np.array(bbox, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    image_pil = Image.fromarray(image_rgb)
                    mask_pil = Image.fromarray(mask).convert("L")
                    result_pil = self.simple_lama(image_pil, mask_pil)
                    result_array = np.array(result_pil)
                    new_image = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
                    
                    self.image_list[idx] = new_image
                    success_count += 1
                    
                except Exception as e:
                    fail_count += 1
                    continue
            
            self.batch_processing = False
            self.root.after(0, lambda: self._finish_batch_processing(success_count, fail_count))
            
        except Exception as e:
            self.batch_processing = False
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _process_batch_image_in_background(self):
        try:
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("Loading LaMa model..."))
                self.simple_lama = SimpleLama()
            
            template = self.template_image
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_edges = cv2.Canny(template_gray, 50, 150)
            
            try:
                threshold = float(self.threshold_var.get())
            except:
                threshold = 0.3
            
            def compute_gradient_direction(img):
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                angle = cv2.phase(gx, gy, angleInDegrees=True)
                return angle
            
            template_grad = compute_gradient_direction(template_gray)
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
            
            total = len(self.image_list)
            success_count = 0
            fail_count = 0
            
            for idx, img in enumerate(self.image_list):
                self.current_image_index = idx
                self.image = self.image_list[idx]
                self.original_image = self.original_image_list[idx]
                self.root.after(0, lambda i=idx: self._update_batch_display(i, total))
                
                try:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_edges = cv2.Canny(img_gray, 50, 150)
                    img_grad = compute_gradient_direction(img_gray)
                    
                    matched_regions = []
                    best_score = 0.0
                    
                    for scale in scales:
                        scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                        scaled_edges = cv2.resize(template_edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                        scaled_grad = cv2.resize(template_grad, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                        sth, stw = scaled_template.shape[:2]
                        
                        if sth >= img_gray.shape[0] or stw >= img_gray.shape[1]:
                            continue
                        
                        result_gray = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                        result_edges = cv2.matchTemplate(img_edges, scaled_edges, cv2.TM_CCOEFF_NORMED)
                        result_grad = cv2.matchTemplate(img_grad, scaled_grad, cv2.TM_CCOEFF_NORMED)
                        result_combined = np.maximum(result_gray, np.maximum(result_edges, result_grad))
                        
                        max_val = np.max(result_combined)
                        if max_val > best_score:
                            best_score = max_val
                        
                        loc = np.where(result_combined >= threshold)
                        for pt in zip(*loc[::-1]):
                            score = result_combined[pt[1], pt[0]]
                            matched_regions.append((pt[0], pt[1], pt[0] + stw, pt[1] + sth, score))
                    
                    if not matched_regions:
                        fail_count += 1
                        continue
                    
                    def overlap(r1, r2):
                        return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])
                    
                    matched_regions.sort(key=lambda x: x[4], reverse=True)
                    filtered_regions = []
                    for region in matched_regions:
                        is_dup = False
                        for fr in filtered_regions:
                            if overlap(region, fr):
                                is_dup = True
                                break
                        if not is_dup:
                            filtered_regions.append(region)
                    
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    for region in filtered_regions:
                        x1, y1, x2, y2 = region[:4]
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    
                    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image_pil = Image.fromarray(image_rgb)
                    mask_pil = Image.fromarray(mask).convert("L")
                    result_pil = self.simple_lama(image_pil, mask_pil)
                    result_array = np.array(result_pil)
                    new_image = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
                    
                    self.image_list[idx] = new_image
                    success_count += 1
                    
                except Exception as e:
                    fail_count += 1
                    continue
            
            self.batch_processing = False
            self.root.after(0, lambda: self._finish_batch_processing(success_count, fail_count))
            
        except Exception as e:
            self.batch_processing = False
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _process_image_in_background(self):
        try:
            self.root.after(0, lambda: self._update_status("Performing multi-scale template matching..."))
            
            template = self.template_image
            img = self.image
            
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            template_edges = cv2.Canny(template_gray, 50, 150)
            img_edges = cv2.Canny(img_gray, 50, 150)
            
            th, tw = template_gray.shape[:2]
            
            try:
                threshold = float(self.threshold_var.get())
            except:
                threshold = 0.3
            
            def compute_gradient_direction(img):
                gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                angle = cv2.phase(gx, gy, angleInDegrees=True)
                return angle
            
            template_grad = compute_gradient_direction(template_gray)
            img_grad = compute_gradient_direction(img_gray)
            
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
            
            matched_regions = []
            best_score = 0.0
            
            for scale in scales:
                scaled_template = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                scaled_edges = cv2.resize(template_edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                scaled_grad = cv2.resize(template_grad, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                sth, stw = scaled_template.shape[:2]
                
                if sth >= img_gray.shape[0] or stw >= img_gray.shape[1]:
                    continue
                
                result_gray = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                result_edges = cv2.matchTemplate(img_edges, scaled_edges, cv2.TM_CCOEFF_NORMED)
                result_grad = cv2.matchTemplate(img_grad, scaled_grad, cv2.TM_CCOEFF_NORMED)
                
                result_combined = np.maximum(result_gray, np.maximum(result_edges, result_grad))
                
                max_val = np.max(result_combined)
                if max_val > best_score:
                    best_score = max_val
                
                loc = np.where(result_combined >= threshold)
                
                for pt in zip(*loc[::-1]):
                    score = result_combined[pt[1], pt[0]]
                    matched_regions.append((pt[0], pt[1], pt[0] + stw, pt[1] + sth, score))
            
            if not matched_regions:
                msg = f"Best match score: {best_score:.2f}. Threshold: {threshold:.2f}. Try lowering threshold."
                self.root.after(0, lambda: self._handle_error(msg))
                return
            
            def overlap(r1, r2):
                return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])
            
            matched_regions.sort(key=lambda x: x[4], reverse=True)
            filtered_regions = []
            for region in matched_regions:
                is_dup = False
                for fr in filtered_regions:
                    if overlap(region, fr):
                        is_dup = True
                        break
                if not is_dup:
                    filtered_regions.append(region)
            
            self.root.after(0, lambda: self._update_status(f"Found {len(filtered_regions)} watermark region(s)"))
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            for region in filtered_regions:
                x1, y1, x2, y2 = region[:4]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            self.root.after(0, lambda: self._update_status("Processing with LaMa AI model..."))
            self._apply_lama_inpaint(mask)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _apply_lama_inpaint(self, mask):
        try:
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("Loading LaMa model..."))
                self.simple_lama = SimpleLama()
            
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
        self.image_list[self.current_image_index] = new_image
        self.roi_selected = False
        self._refresh_display("Watermark removed! Click [Save] to save.")
    
    def _finish_batch_processing(self, success_count, fail_count):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        
        self.current_image_index = 0
        self.image = self.image_list[0]
        self.original_image = self.original_image_list[0]
        self.roi_selected = False
        self.zoom_scale = 1.0
        
        self._update_nav_buttons()
        
        msg = f"Batch complete: {success_count} succeeded, {fail_count} failed"
        self._update_display(msg)
    
    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._update_status(f"Error: {error_msg}")
        
    def _save_image(self):
        if self.image is None:
            self._update_status("No image to save!")
            return
        
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            self._update_status("No original path found!")
            return
        
        current_path = self.image_paths[self.current_image_index]
        if current_path:
            try:
                ext = current_path.rsplit('.', 1)[-1]
                if ext.lower() in ['jpg', 'jpeg']:
                    cv2.imencode('.jpg', self.image)[1].tofile(current_path)
                elif ext.lower() == 'png':
                    cv2.imencode('.png', self.image)[1].tofile(current_path)
                elif ext.lower() == 'webp':
                    cv2.imencode('.webp', self.image)[1].tofile(current_path)
                else:
                    cv2.imencode('.png', self.image)[1].tofile(current_path)
                self._update_status(f"Saved: {current_path}")
            except Exception as e:
                self._update_status(f"Failed to save: {str(e)}")
        else:
            self._update_status("No original path for this image!")
    
    def _batch_save_image(self):
        if not self.image_list:
            self._update_status("No images to save!")
            return
        
        saved_count = 0
        failed_count = 0
        for idx, img in enumerate(self.image_list):
            if idx < len(self.image_paths) and self.image_paths[idx]:
                try:
                    current_path = self.image_paths[idx]
                    ext = current_path.rsplit('.', 1)[-1]
                    if ext.lower() in ['jpg', 'jpeg']:
                        cv2.imencode('.jpg', img)[1].tofile(current_path)
                    elif ext.lower() == 'png':
                        cv2.imencode('.png', img)[1].tofile(current_path)
                    elif ext.lower() == 'webp':
                        cv2.imencode('.webp', img)[1].tofile(current_path)
                    else:
                        cv2.imencode('.png', img)[1].tofile(current_path)
                    saved_count += 1
                except Exception as e:
                    failed_count += 1
            else:
                failed_count += 1
        
        if failed_count == 0:
            self._update_status(f"Batch saved {saved_count} images to original locations")
        else:
            self._update_status(f"Saved {saved_count}, failed {failed_count}")
            
    def _reset_image(self):
        if self.original_image_list:
            for idx, orig in enumerate(self.original_image_list):
                self.image_list[idx] = orig.copy()
            
            self.image = self.image_list[self.current_image_index]
            self.original_image = self.original_image_list[self.current_image_index]
            self.roi_selected = False
            self.zoom_scale = 1.0
            
            msg = f"Reset image {self.current_image_index + 1}/{len(self.image_list)}. "
            if self.mode == "text":
                msg += "Enter watermark text."
            elif self.mode == "image":
                msg += "Load template and click Remove." if self.template_image is None else "Click Remove to find watermark."
            else:
                msg += "Drag to select watermark area."
            self._update_display(msg)
            
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
        if self.image is None or self.mode != "box":
            return
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        self.fx, self.fy = event.x, event.y
        
    def _on_mouse_move(self, event):
        if not self.drawing or self.image is None or self.mode != "box":
            return
        self.fx, self.fy = event.x, event.y
        self._draw_selection()
        
    def _on_mouse_up(self, event):
        if not self.drawing or self.image is None or self.mode != "box":
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
