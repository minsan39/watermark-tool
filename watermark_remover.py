import cv2
import numpy as np
from tkinter import filedialog, Button, Canvas, Frame, Label, Entry, StringVar, Toplevel
from tkinter import ttk
from PIL import Image, ImageTk
from simple_lama_inpainting import SimpleLama
import threading
from rapidocr_onnxruntime import RapidOCR
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tkinterdnd2 import TkinterDnD, DND_FILES

class WatermarkRemover:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Remover")
        
        self.device = self._detect_device()
        
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
    
    def _detect_device(self):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {device_name}")
            return "cuda"
        else:
            print("No GPU detected, using CPU")
            return "cpu"
    
    def _setup_styles(self):
        self.root.configure(bg="#f5f5f5")
        self.button_styles = {
            "font": ("Microsoft YaHei UI", 10),
            "padding": 8,
            "radius": 8
        }
    
    def _create_rounded_button(self, parent, text, command, bg_color, fg_color, state="normal"):
        btn_frame = Frame(parent, bg=parent["bg"])
        btn_frame.pack(side="left", padx=4)
        
        canvas = Canvas(btn_frame, width=80, height=32, 
                       highlightthickness=0, bg=parent["bg"])
        canvas.pack()
        
        def draw_button(event=None):
            canvas.delete("all")
            w, h = 80, 32
            r = self.button_styles["radius"]
            
            if btn_state["disabled"]:
                fill_color = "#cccccc"
                text_color = "#888888"
            elif btn_state["hover"]:
                self._darken_color(bg_color, 0.15)
                fill_color = self._darken_color(bg_color, 0.1)
                text_color = fg_color
            else:
                fill_color = bg_color
                text_color = fg_color
            
            canvas.create_arc(0, 0, r*2, r*2, start=90, extent=90, 
                            fill=fill_color, outline=fill_color)
            canvas.create_arc(w-r*2, 0, w, r*2, start=0, extent=90, 
                            fill=fill_color, outline=fill_color)
            canvas.create_arc(0, h-r*2, r*2, h, start=180, extent=90, 
                            fill=fill_color, outline=fill_color)
            canvas.create_arc(w-r*2, h-r*2, w, h, start=270, extent=90, 
                            fill=fill_color, outline=fill_color)
            
            canvas.create_rectangle(r, 0, w-r, h, fill=fill_color, outline=fill_color)
            canvas.create_rectangle(0, r, w, h-r, fill=fill_color, outline=fill_color)
            
            canvas.create_text(w//2, h//2, text=text, fill=text_color, 
                             font=self.button_styles["font"])
        
        btn_state = {"hover": False, "disabled": state == "disabled"}
        
        def on_enter(e):
            if not btn_state["disabled"]:
                btn_state["hover"] = True
                draw_button()
        
        def on_leave(e):
            btn_state["hover"] = False
            draw_button()
        
        def on_click(e):
            if not btn_state["disabled"]:
                command()
        
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
        canvas.bind("<Button-1>", on_click)
        
        draw_button()
        
        def set_state(new_state):
            btn_state["disabled"] = (new_state == "disabled")
            draw_button()
        
        canvas.set_state = set_state
        return canvas
    
    def _darken_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(r * (1 - factor))
        g = int(g * (1 - factor))
        b = int(b * (1 - factor))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _setup_ui(self):
        self._setup_styles()
        
        top_frame = Frame(self.root, bg="#f5f5f5")
        top_frame.pack(fill="x", padx=10, pady=10)
        
        left_frame = Frame(top_frame, bg="#f5f5f5")
        left_frame.pack(side="left")
        
        right_frame = Frame(top_frame, bg="#f5f5f5")
        right_frame.pack(side="right")
        
        self._create_rounded_button(left_frame, "打开", self._open_image, "#4a90d9", "white")
        self._create_rounded_button(left_frame, "重置", self._reset_image, "#6c757d", "white")
        
        self._create_rounded_button(right_frame, "移除", self._remove_watermark, "#28a745", "white")
        self.btn_batch_remove = self._create_rounded_button(right_frame, "批量移除", self._batch_remove_watermark, "#20c997", "white", state="disabled")
        self._create_rounded_button(right_frame, "保存", self._save_image, "#007bff", "white")
        self.btn_batch_save = self._create_rounded_button(right_frame, "保存所有", self._batch_save_image, "#17a2b8", "white", state="disabled")
        
        self.tab_frame = Frame(self.root, bg="#f5f5f5")
        self.tab_frame.pack(fill="x", padx=10, pady=5)
        
        self.tab_canvas = Canvas(self.tab_frame, height=35, bg="#e8e8e8", highlightthickness=0)
        self.tab_canvas.pack(side="left", fill="x", expand=True)
        
        self.tab_inner_frame = Frame(self.tab_canvas, bg="#e8e8e8")
        self.tab_canvas.create_window(0, 0, window=self.tab_inner_frame, anchor="nw")
        
        self.tab_canvas.bind("<MouseWheel>", self._on_tab_scroll)
        self.tab_canvas.bind("<Button-4>", self._on_tab_scroll)
        self.tab_canvas.bind("<Button-5>", self._on_tab_scroll)
        self.tab_inner_frame.bind("<MouseWheel>", self._on_tab_scroll)
        
        self.tabs = []
        self.close_buttons = []
        self.tab_hover_id = None
        
        self.drag_data = {
            "dragging": False,
            "drag_tab_idx": -1,
            "start_x": 0,
            "drag_text": ""
        }
        self.insert_indicator = None
        self.drag_preview = None
        
        mode_frame = Frame(self.root, bg="#f5f5f5")
        mode_frame.pack(pady=5, fill="x", padx=10)
        
        self.mode_var = StringVar(value="box")
        Label(mode_frame, text="模式:", bg="#f5f5f5", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=5)
        
        style = ttk.Style()
        style.configure("Mode.TRadiobutton", background="#f5f5f5", font=("Microsoft YaHei UI", 9))
        
        ttk.Radiobutton(mode_frame, text="框选", variable=self.mode_var, 
                        value="box", command=self._switch_mode, style="Mode.TRadiobutton").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="文字", variable=self.mode_var, 
                        value="text", command=self._switch_mode, style="Mode.TRadiobutton").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="图像", variable=self.mode_var, 
                        value="image", command=self._switch_mode, style="Mode.TRadiobutton").pack(side="left", padx=5)
        
        self.text_frame = Frame(self.root, bg="#f5f5f5")
        Label(self.text_frame, text="水印文字:", bg="#f5f5f5", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=5)
        self.text_entry = Entry(self.text_frame, width=40, font=("Microsoft YaHei UI", 9), relief="solid", bd=1)
        self.text_entry.pack(side="left", padx=5)
        self.text_frame.pack(pady=5, fill="x", padx=10)
        self.text_frame.pack_forget()
        
        self.image_frame = Frame(self.root, bg="#f5f5f5")
        Label(self.image_frame, text="模板:", bg="#f5f5f5", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=5)
        self._create_rounded_button(self.image_frame, "加载模板", self._load_template, "#6c757d", "white")
        self.template_label = Label(self.image_frame, text="未加载模板", fg="gray", bg="#f5f5f5", font=("Microsoft YaHei UI", 9))
        self.template_label.pack(side="left", padx=5)
        
        self.threshold_frame = Frame(self.root, bg="#f5f5f5")
        Label(self.threshold_frame, text="阈值:", bg="#f5f5f5", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=5)
        self.threshold_var = StringVar(value="0.3")
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0.1, to=0.9, variable=self.threshold_var, orient="horizontal", length=150)
        self.threshold_scale.pack(side="left", padx=5)
        self.threshold_label = Label(self.threshold_frame, text="0.30", width=5, bg="#f5f5f5", font=("Microsoft YaHei UI", 9))
        self.threshold_label.pack(side="left", padx=5)
        self.threshold_scale.bind("<Motion>", self._update_threshold_label)
        self.threshold_scale.bind("<ButtonRelease-1>", self._update_threshold_label)
        self.threshold_frame.pack(pady=5, fill="x", padx=10)
        self.threshold_frame.pack_forget()
        self.image_frame.pack(pady=5, fill="x", padx=10)
        self.image_frame.pack_forget()
        
        canvas_frame = Frame(self.root, bg="#f5f5f5")
        canvas_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.canvas = Canvas(canvas_frame, width=self.canvas_w, height=self.canvas_h, bg="#3c3c3c", highlightthickness=1, highlightbackground="#d0d0d0")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, text="点击 [打开] 或拖拽图片到这里", fill="#888888", font=("Microsoft YaHei UI", 14))
        
        self.canvas.drop_target_register(DND_FILES)
        self.canvas.dnd_bind('<<Drop>>', self._on_drop)
        
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)
        
        self.status_label = Frame(self.root, bd=0, bg="#e0e0e0")
        self.status_label.pack(fill="x", padx=10, pady=5)
        self.status_text = Label(self.status_label, text="就绪 | 滚轮缩放, 右键拖拽平移", anchor="w", bg="#e0e0e0", font=("Microsoft YaHei UI", 9), padx=10, pady=5)
        self.status_text.pack(fill="x")
        
        self.progress_frame = Frame(self.root, bg="#f5f5f5")
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
    
    def _update_tabs(self):
        for tab in self.tabs:
            tab.destroy()
        for btn in self.close_buttons:
            btn.destroy()
        self.tabs = []
        self.close_buttons = []
        
        if self.insert_indicator:
            self.tab_canvas.delete("insert_indicator")
            self.insert_indicator = None
        
        for idx, path in enumerate(self.image_paths):
            filename = path.split('\\')[-1].split('/')[-1]
            if len(filename) > 15:
                filename = filename[:12] + "..."
            
            is_active = (idx == self.current_image_index)
            bg_color = "#ffffff" if is_active else "#e0e0e0"
            fg_color = "#000000"
            
            tab_container = Frame(self.tab_inner_frame, bg=bg_color, cursor="hand2")
            tab_container.pack(side="left", padx=2, pady=2)
            
            tab_label = Label(tab_container, text=filename, bg=bg_color, fg=fg_color, 
                             padx=8, pady=5, cursor="hand2")
            tab_label.pack(side="left")
            
            close_btn = Label(tab_container, text="✕", bg=bg_color, fg="#888888",
                             padx=5, pady=5, cursor="hand2")
            close_btn.pack(side="left")
            
            tab_label.bind("<Button-1>", lambda e, i=idx: self._on_tab_click(e, i))
            tab_container.bind("<Button-1>", lambda e, i=idx: self._on_tab_click(e, i))
            
            tab_label.bind("<B1-Motion>", lambda e, i=idx: self._on_tab_drag(e, i))
            tab_container.bind("<B1-Motion>", lambda e, i=idx: self._on_tab_drag(e, i))
            
            tab_label.bind("<ButtonRelease-1>", lambda e, i=idx: self._on_tab_release(e, i))
            tab_container.bind("<ButtonRelease-1>", lambda e, i=idx: self._on_tab_release(e, i))
            
            close_btn.bind("<Button-1>", lambda e, i=idx: self._close_tab(i))
            close_btn.bind("<Enter>", lambda e, btn=close_btn: btn.config(fg="#ff4444"))
            close_btn.bind("<Leave>", lambda e, btn=close_btn: btn.config(fg="#888888"))
            
            tab_container.bind("<Enter>", lambda e, c=tab_container, l=tab_label, b=close_btn, active=is_active: 
                              self._on_tab_enter(c, l, b, active))
            tab_container.bind("<Leave>", lambda e, c=tab_container, l=tab_label, b=close_btn, active=is_active: 
                              self._on_tab_leave(c, l, b, active))
            
            for widget in [tab_container, tab_label, close_btn]:
                widget.bind("<MouseWheel>", self._on_tab_scroll)
                widget.bind("<Button-4>", self._on_tab_scroll)
                widget.bind("<Button-5>", self._on_tab_scroll)
            
            self.tabs.append(tab_container)
            self.close_buttons.append(close_btn)
        
        self.root.after(10, self._update_tab_scroll_region)
        
        total = len(self.image_list)
        if total > 1:
            self.btn_batch_remove.set_state("normal")
            self.btn_batch_save.set_state("normal")
        else:
            self.btn_batch_remove.set_state("disabled")
            self.btn_batch_save.set_state("disabled")
    
    def _on_tab_click(self, event, idx):
        self.drag_data["dragging"] = False
        self.drag_data["drag_tab_idx"] = idx
        self.drag_data["start_x"] = event.x_root
    
    def _on_tab_drag(self, event, idx):
        if abs(event.x_root - self.drag_data["start_x"]) < 10:
            return
        
        if not self.drag_data["dragging"]:
            self.drag_data["dragging"] = True
            self.drag_data["drag_text"] = self.image_paths[idx].split('\\')[-1].split('/')[-1]
            if len(self.drag_data["drag_text"]) > 15:
                self.drag_data["drag_text"] = self.drag_data["drag_text"][:12] + "..."
            self._show_drag_indicator(idx)
            self._create_drag_preview(event)
        
        self._update_drag_preview(event)
        self._update_insert_position(event)
    
    def _show_drag_indicator(self, drag_idx):
        if drag_idx < len(self.tabs):
            tab = self.tabs[drag_idx]
            tab.config(bg="#b8d4e8")
            for child in tab.winfo_children():
                child.config(bg="#b8d4e8")
    
    def _create_drag_preview(self, event):
        if self.drag_preview:
            self.drag_preview.destroy()
        
        self.drag_preview = Toplevel(self.root)
        self.drag_preview.overrideredirect(True)
        self.drag_preview.attributes('-topmost', True)
        self.drag_preview.attributes('-alpha', 0.85)
        
        preview_frame = Frame(self.drag_preview, bg="#0078d4", bd=1, relief="solid")
        preview_frame.pack(fill="both", expand=True)
        
        label = Label(preview_frame, text=self.drag_data["drag_text"], 
                     bg="#0078d4", fg="white", padx=10, pady=5, font=("Arial", 9))
        label.pack()
        
        self._position_drag_preview(event)
    
    def _position_drag_preview(self, event):
        if self.drag_preview:
            x = event.x_root - 40
            y = event.y_root - 15
            self.drag_preview.geometry(f"+{x}+{y}")
    
    def _update_drag_preview(self, event):
        self._position_drag_preview(event)
    
    def _update_insert_position(self, event):
        if self.insert_indicator:
            self.tab_canvas.delete("insert_indicator")
        
        canvas_x = self.tab_canvas.winfo_rootx()
        mouse_x = event.x_root - canvas_x
        
        insert_idx = self._get_insert_index(mouse_x)
        
        if insert_idx is not None and insert_idx <= len(self.tabs):
            if insert_idx < len(self.tabs):
                tab = self.tabs[insert_idx]
                tab_x = tab.winfo_x()
                self.insert_indicator = self.tab_canvas.create_line(
                    tab_x - 2, 2, tab_x - 2, 33, 
                    fill="#0078d4", width=3, tags="insert_indicator"
                )
            else:
                if self.tabs:
                    last_tab = self.tabs[-1]
                    last_x = last_tab.winfo_x() + last_tab.winfo_width()
                    self.insert_indicator = self.tab_canvas.create_line(
                        last_x + 2, 2, last_x + 2, 33, 
                        fill="#0078d4", width=3, tags="insert_indicator"
                    )
    
    def _get_insert_index(self, mouse_x):
        for idx, tab in enumerate(self.tabs):
            tab_x = tab.winfo_x()
            tab_w = tab.winfo_width()
            tab_center = tab_x + tab_w // 2
            
            if mouse_x < tab_center:
                return idx
        return len(self.tabs)
    
    def _on_tab_release(self, event, drag_idx):
        if self.insert_indicator:
            self.tab_canvas.delete("insert_indicator")
            self.insert_indicator = None
        
        if self.drag_preview:
            self.drag_preview.destroy()
            self.drag_preview = None
        
        if self.drag_data["dragging"]:
            canvas_x = self.tab_canvas.winfo_rootx()
            mouse_x = event.x_root - canvas_x
            insert_idx = self._get_insert_index(mouse_x)
            
            if insert_idx is not None and insert_idx != drag_idx:
                self._reorder_tabs(drag_idx, insert_idx)
            else:
                self._update_tabs()
        else:
            self._switch_to_tab(drag_idx)
        
        self.drag_data["dragging"] = False
        self.drag_data["drag_tab_idx"] = -1
        self.drag_data["drag_text"] = ""
    
    def _reorder_tabs(self, from_idx, to_idx):
        if from_idx < to_idx:
            to_idx -= 1
        
        img = self.image_list.pop(from_idx)
        self.image_list.insert(to_idx, img)
        
        orig = self.original_image_list.pop(from_idx)
        self.original_image_list.insert(to_idx, orig)
        
        path = self.image_paths.pop(from_idx)
        self.image_paths.insert(to_idx, path)
        
        if self.current_image_index == from_idx:
            self.current_image_index = to_idx
        elif from_idx < self.current_image_index <= to_idx:
            self.current_image_index -= 1
        elif to_idx <= self.current_image_index < from_idx:
            self.current_image_index += 1
        
        self._update_tabs()
    
    def _on_tab_enter(self, container, label, btn, is_active):
        if not is_active:
            container.config(bg="#d0d0d0")
            label.config(bg="#d0d0d0")
            btn.config(bg="#d0d0d0")
    
    def _on_tab_leave(self, container, label, btn, is_active):
        if is_active:
            container.config(bg="#ffffff")
            label.config(bg="#ffffff")
            btn.config(bg="#ffffff")
        else:
            container.config(bg="#e0e0e0")
            label.config(bg="#e0e0e0")
            btn.config(bg="#e0e0e0")
    
    def _update_tab_scroll_region(self):
        self.tab_canvas.update_idletasks()
        self.tab_canvas.config(scrollregion=self.tab_canvas.bbox("all"))
    
    def _on_tab_scroll(self, event):
        scroll_amount = 3
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            self.tab_canvas.xview_scroll(-scroll_amount, "units")
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            self.tab_canvas.xview_scroll(scroll_amount, "units")
    
    def _switch_to_tab(self, idx):
        if 0 <= idx < len(self.image_list):
            self.current_image_index = idx
            self.image = self.image_list[idx]
            self.original_image = self.original_image_list[idx]
            self.roi_selected = False
            self.zoom_scale = 1.0
            self._update_tabs()
            self._update_display("")
    
    def _close_tab(self, idx):
        if len(self.image_list) <= 1:
            self.image_list = []
            self.original_image_list = []
            self.image_paths = []
            self.current_image_index = -1
            self.image = None
            self.original_image = None
            self._update_tabs()
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, 
                                   text="点击 [打开] 或拖拽图片到这里", 
                                   fill="#888888", font=("Microsoft YaHei UI", 14))
            self._update_status("就绪 | 滚轮缩放, 右键拖拽平移")
            return
        
        del self.image_list[idx]
        del self.original_image_list[idx]
        del self.image_paths[idx]
        
        if self.current_image_index >= len(self.image_list):
            self.current_image_index = len(self.image_list) - 1
        elif self.current_image_index > idx:
            self.current_image_index -= 1
        
        if self.image_list:
            self.image = self.image_list[self.current_image_index]
            self.original_image = self.original_image_list[self.current_image_index]
        else:
            self.image = None
            self.original_image = None
        
        self.roi_selected = False
        self._update_tabs()
        
        if self.image is not None:
            self._update_display("")
        else:
            self.canvas.delete("all")
            self.canvas.create_text(self.canvas_w//2, self.canvas_h//2, 
                                   text="点击 [打开] 或拖拽图片到这里", 
                                   fill="#888888", font=("Microsoft YaHei UI", 14))
    
    def _update_batch_display(self, current_idx, total):
        self._update_tabs()
        self._refresh_display(f"处理中 {current_idx + 1}/{total}...")
    
    def _switch_mode(self):
        self.mode = self.mode_var.get()
        self.roi_selected = False
        
        self.text_frame.pack_forget()
        self.image_frame.pack_forget()
        self.threshold_frame.pack_forget()
        
        if self.mode == "text":
            self.text_frame.pack(pady=5, fill="x", padx=10)
            self._update_status("文字模式: 输入水印文字后点击 [移除]")
        elif self.mode == "image":
            self.image_frame.pack(pady=5, fill="x", padx=10)
            self.threshold_frame.pack(pady=5, fill="x", padx=10)
            if self.template_image is not None:
                self._update_status("图像模式: 点击 [移除] 查找并移除水印")
            else:
                self._update_status("图像模式: 请先加载模板图像")
        else:
            self._update_status("框选模式: 拖拽选择水印区域")
        
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
                    self._update_status("模板图像解码失败!")
            except Exception as e:
                self._update_status(f"错误: {str(e)}")
    
    def _show_template_crop_window(self, full_template):
        crop_window = Toplevel(self.root)
        crop_window.title("选择水印区域")
        
        h, w = full_template.shape[:2]
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        display_w = int(w * scale)
        display_h = int(h * scale)
        
        crop_window.geometry(f"{display_w + 20}x{display_h + 100}")
        
        Label(crop_window, text="拖拽选择水印区域，然后点击确认").pack(pady=5)
        
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
                    self.template_label.config(text=f"{tw}x{th} 已裁剪", fg="green")
                    self._update_status("模板已裁剪。点击 [移除] 查找并移除水印。")
                    crop_window.destroy()
                else:
                    self._update_status("选择区域无效!")
            else:
                self._update_status("请先选择一个区域!")
        
        btn_frame = Frame(crop_window)
        btn_frame.pack(pady=10)
        Button(btn_frame, text="确认", command=confirm_crop, width=10).pack(side="left", padx=5)
        Button(btn_frame, text="取消", command=crop_window.destroy, width=10).pack(side="left", padx=5)
        
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
                count_msg = f"已添加 {new_count} 张图片, 共 {total_count} 张"
                if failed_files:
                    count_msg += f", {len(failed_files)} 张失败"
                
                self._update_tabs()
                msg = f"{count_msg}. "
                if self.mode == "text":
                    msg += "请输入水印文字。"
                elif self.mode == "image":
                    msg += "加载模板后点击移除。" if self.template_image is None else "点击移除查找水印。"
                else:
                    msg += "拖拽选择水印区域。"
                self._update_display(msg)
            else:
                self._update_status("没有加载有效的图片!")
    
    def _on_drop(self, event):
        dropped_data = event.data
        
        if dropped_data.startswith('{'):
            paths = []
            current_path = ""
            in_braces = False
            for char in dropped_data:
                if char == '{':
                    in_braces = True
                elif char == '}':
                    in_braces = False
                    if current_path:
                        paths.append(current_path)
                        current_path = ""
                elif char == ' ' and not in_braces:
                    if current_path:
                        paths.append(current_path)
                        current_path = ""
                else:
                    current_path += char
            if current_path:
                paths.append(current_path)
        else:
            paths = dropped_data.split()
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_paths = []
        for path in paths:
            path = path.strip()
            if path.lower().endswith(valid_extensions):
                try:
                    with open(path, 'rb') as f:
                        img_array = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        image_paths.append((path, img))
                except:
                    pass
        
        if not image_paths:
            self._update_status("拖入的文件中没有有效图片!")
            return
        
        for path, img in image_paths:
            self.image_list.append(img)
            self.original_image_list.append(img.copy())
            self.image_paths.append(path)
        
        self.current_image_index = len(self.image_list) - 1
        self.image = self.image_list[self.current_image_index]
        self.original_image = self.original_image_list[self.current_image_index]
        self.roi_selected = False
        self.zoom_scale = 1.0
        
        self._update_tabs()
        msg = f"已拖入 {len(image_paths)} 张图片。"
        if self.mode == "text":
            msg += "请输入水印文字。"
        elif self.mode == "image":
            msg += "加载模板后点击移除。" if self.template_image is None else "点击移除查找水印。"
        else:
            msg += "拖拽选择水印区域。"
        self._update_display(msg)
                
    def _remove_watermark(self):
        if self.image is None:
            self._update_status("请先打开图片!")
            return
        
        if self.mode == "box":
            if not self.roi_selected:
                self._update_status("请先选择水印区域!")
                return
            self._start_box_removal()
        elif self.mode == "text":
            watermark_text = self.text_entry.get().strip()
            if not watermark_text:
                self._update_status("请输入水印文字!")
                return
            self._start_text_removal(watermark_text)
        else:
            if self.template_image is None:
                self._update_status("请先加载模板图像!")
                return
            self._start_image_removal()
    
    def _batch_remove_watermark(self):
        if self.image is None:
            self._update_status("请先打开图片!")
            return
        
        if len(self.image_list) <= 1:
            self._update_status("批量移除需要多张图片!")
            return
        
        if self.mode == "box":
            self._update_status("框选模式不支持批量处理。请使用文字或图像模式。")
            return
        elif self.mode == "text":
            watermark_text = self.text_entry.get().strip()
            if not watermark_text:
                self._update_status("请输入水印文字!")
                return
            self._start_batch_text_removal(watermark_text)
        else:
            if self.template_image is None:
                self._update_status("请先加载模板图像!")
                return
            self._start_batch_image_removal()
    
    def _start_box_removal(self):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("正在使用 LaMa AI 模型处理...")
        self.root.update()
        
        threading.Thread(target=self._process_box_in_background, daemon=True).start()
    
    def _start_text_removal(self, watermark_text):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("正在初始化 OCR 并检测文字...")
        self.root.update()
        
        threading.Thread(target=self._process_text_in_background, args=(watermark_text,), daemon=True).start()
    
    def _start_image_removal(self):
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status("正在搜索水印模式...")
        self.root.update()
        
        threading.Thread(target=self._process_image_in_background, daemon=True).start()
    
    def _start_batch_text_removal(self, watermark_text):
        self.batch_processing = True
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status(f"批量处理: 0/{len(self.image_list)} 张图片...")
        self.root.update()
        
        threading.Thread(target=self._process_batch_text_in_background, args=(watermark_text,), daemon=True).start()
    
    def _start_batch_image_removal(self):
        self.batch_processing = True
        self.progress_bar.pack(fill="x")
        self.progress_bar.start(10)
        self._update_status(f"批量处理: 0/{len(self.image_list)} 张图片...")
        self.root.update()
        
        threading.Thread(target=self._process_batch_image_in_background, daemon=True).start()
    
    def _process_box_in_background(self):
        try:
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("正在加载 LaMa 模型 (首次需下载约200MB)..."))
                self.simple_lama = SimpleLama(device=torch.device(self.device))
                self.root.after(0, lambda: self._update_status("正在使用 LaMa AI 模型处理..."))
                
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
                self.root.after(0, lambda: self._update_status("正在加载 OCR 模型..."))
                self.ocr_reader = RapidOCR()
                self.root.after(0, lambda: self._update_status("正在检测图片中的文字..."))
            
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
                    self.root.after(0, lambda t=text: self._update_status(f"找到匹配文字: '{t}'"))
            
            if not matched_boxes:
                self.root.after(0, lambda: self._handle_error(f"图片中未找到文字 '{watermark_text}'"))
                return
            
            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            
            for bbox in matched_boxes:
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            self.root.after(0, lambda: self._update_status("正在使用 LaMa AI 模型处理..."))
            self._apply_lama_inpaint(mask)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _process_batch_text_in_background(self, watermark_text):
        try:
            if self.ocr_reader is None:
                self.root.after(0, lambda: self._update_status("正在加载 OCR 模型..."))
                self.ocr_reader = RapidOCR()
            
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("正在加载 LaMa 模型..."))
                self.simple_lama = SimpleLama(device=torch.device(self.device))
            
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
                self.root.after(0, lambda: self._update_status("正在加载 LaMa 模型..."))
                self.simple_lama = SimpleLama(device=torch.device(self.device))
            
            template = self.template_image
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_edges = cv2.Canny(template_gray, 50, 150)
            
            gx = cv2.Sobel(template_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(template_gray, cv2.CV_32F, 0, 1, ksize=3)
            template_grad = cv2.phase(gx, gy, angleInDegrees=True)
            
            try:
                threshold = float(self.threshold_var.get())
            except:
                threshold = 0.3
            
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
                    
                    gx_img = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
                    gy_img = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
                    img_grad = cv2.phase(gx_img, gy_img, angleInDegrees=True)
                    
                    img_h, img_w = img_gray.shape
                    
                    scaled_templates = {}
                    for scale in scales:
                        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                        t_gray = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=interp)
                        t_edges = cv2.resize(template_edges, None, fx=scale, fy=scale, interpolation=interp)
                        t_grad = cv2.resize(template_grad, None, fx=scale, fy=scale, interpolation=interp)
                        sth, stw = t_gray.shape[:2]
                        if sth < img_h and stw < img_w:
                            scaled_templates[scale] = (t_gray, t_edges, t_grad, sth, stw)
                    
                    def match_at_scale(scale_data):
                        scale, (t_gray, t_edges, t_grad, sth, stw) = scale_data
                        results = []
                        
                        result_gray = cv2.matchTemplate(img_gray, t_gray, cv2.TM_CCOEFF_NORMED)
                        result_edges = cv2.matchTemplate(img_edges, t_edges, cv2.TM_CCOEFF_NORMED)
                        result_grad = cv2.matchTemplate(img_grad, t_grad, cv2.TM_CCOEFF_NORMED)
                        
                        result_combined = np.maximum(result_gray, np.maximum(result_edges, result_grad))
                        max_val = np.max(result_combined)
                        
                        if max_val >= threshold:
                            loc = np.where(result_combined >= threshold)
                            for pt in zip(*loc[::-1]):
                                score = result_combined[pt[1], pt[0]]
                                results.append((pt[0], pt[1], pt[0] + stw, pt[1] + sth, score))
                        
                        return max_val, results
                    
                    matched_regions = []
                    best_score = 0.0
                    
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = {executor.submit(match_at_scale, (s, t)): s for s, t in scaled_templates.items()}
                        for future in as_completed(futures):
                            try:
                                max_val, regions = future.result()
                                if max_val > best_score:
                                    best_score = max_val
                                matched_regions.extend(regions)
                            except Exception:
                                pass
                    
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
            self.root.after(0, lambda: self._update_status("正在执行并行多尺度模板匹配..."))
            
            template = self.template_image
            img = self.image
            
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            template_edges = cv2.Canny(template_gray, 50, 150)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 50, 150)
            
            gx = cv2.Sobel(template_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(template_gray, cv2.CV_32F, 0, 1, ksize=3)
            template_grad = cv2.phase(gx, gy, angleInDegrees=True)
            
            gx_img = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy_img = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
            img_grad = cv2.phase(gx_img, gy_img, angleInDegrees=True)
            
            try:
                threshold = float(self.threshold_var.get())
            except:
                threshold = 0.3
            
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
            
            img_h, img_w = img_gray.shape
            
            scaled_templates = {}
            for scale in scales:
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
                t_gray = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=interp)
                t_edges = cv2.resize(template_edges, None, fx=scale, fy=scale, interpolation=interp)
                t_grad = cv2.resize(template_grad, None, fx=scale, fy=scale, interpolation=interp)
                sth, stw = t_gray.shape[:2]
                if sth < img_h and stw < img_w:
                    scaled_templates[scale] = (t_gray, t_edges, t_grad, sth, stw)
            
            def match_at_scale(scale_data):
                scale, (t_gray, t_edges, t_grad, sth, stw) = scale_data
                results = []
                
                result_gray = cv2.matchTemplate(img_gray, t_gray, cv2.TM_CCOEFF_NORMED)
                result_edges = cv2.matchTemplate(img_edges, t_edges, cv2.TM_CCOEFF_NORMED)
                result_grad = cv2.matchTemplate(img_grad, t_grad, cv2.TM_CCOEFF_NORMED)
                
                result_combined = np.maximum(result_gray, np.maximum(result_edges, result_grad))
                max_val = np.max(result_combined)
                
                if max_val >= threshold:
                    loc = np.where(result_combined >= threshold)
                    for pt in zip(*loc[::-1]):
                        score = result_combined[pt[1], pt[0]]
                        results.append((pt[0], pt[1], pt[0] + stw, pt[1] + sth, score))
                
                return max_val, results
            
            matched_regions = []
            best_score = 0.0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(match_at_scale, (s, t)): s for s, t in scaled_templates.items()}
                for future in as_completed(futures):
                    try:
                        max_val, regions = future.result()
                        if max_val > best_score:
                            best_score = max_val
                        matched_regions.extend(regions)
                    except Exception:
                        pass
            
            if not matched_regions:
                msg = f"最佳匹配分数: {best_score:.2f}。阈值: {threshold:.2f}。请尝试降低阈值。"
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
            
            self.root.after(0, lambda: self._update_status(f"找到 {len(filtered_regions)} 个水印区域"))
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            for region in filtered_regions:
                x1, y1, x2, y2 = region[:4]
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
            self.root.after(0, lambda: self._update_status("正在使用 LaMa AI 模型处理..."))
            self._apply_lama_inpaint(mask)
            
        except Exception as e:
            self.root.after(0, lambda: self._handle_error(str(e)))
    
    def _apply_lama_inpaint(self, mask):
        try:
            if self.simple_lama is None:
                self.root.after(0, lambda: self._update_status("正在加载 LaMa 模型..."))
                self.simple_lama = SimpleLama(device=torch.device(self.device))
            
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
        self._refresh_display("水印已移除! 点击 [保存] 保存图片。")
    
    def _finish_batch_processing(self, success_count, fail_count):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        
        self.current_image_index = 0
        self.image = self.image_list[0]
        self.original_image = self.original_image_list[0]
        self.roi_selected = False
        self.zoom_scale = 1.0
        
        self._update_tabs()
        
        msg = f"批量完成: {success_count} 张成功, {fail_count} 张失败"
        self._update_display(msg)
    
    def _handle_error(self, error_msg):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self._update_status(f"错误: {error_msg}")
        
    def _save_image(self):
        if self.image is None:
            self._update_status("没有可保存的图片!")
            return
        
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            self._update_status("未找到原始路径!")
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
                self._update_status(f"已保存: {current_path}")
            except Exception as e:
                self._update_status(f"保存失败: {str(e)}")
        else:
            self._update_status("此图片没有原始路径!")
    
    def _batch_save_image(self):
        if not self.image_list:
            self._update_status("没有可保存的图片!")
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
            self._update_status(f"已批量保存 {saved_count} 张图片到原位置")
        else:
            self._update_status(f"保存 {saved_count} 张成功, {failed_count} 张失败")
            
    def _reset_image(self):
        if self.original_image_list:
            for idx, orig in enumerate(self.original_image_list):
                self.image_list[idx] = orig.copy()
            
            self.image = self.image_list[self.current_image_index]
            self.original_image = self.original_image_list[self.current_image_index]
            self.roi_selected = False
            self.zoom_scale = 1.0
            
            msg = f"已重置图片 {self.current_image_index + 1}/{len(self.image_list)}。"
            if self.mode == "text":
                msg += "请输入水印文字。"
            elif self.mode == "image":
                msg += "加载模板后点击移除。" if self.template_image is None else "点击移除查找水印。"
            else:
                msg += "拖拽选择水印区域。"
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
            self._update_status(f"{message} | 缩放: {zoom_pct}%")
        else:
            self._update_status(f"缩放: {zoom_pct}%")
            
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
        self._update_status("区域已选择。点击 [移除] 移除水印。")
        
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
        self._update_status(f"缩放: {zoom_pct}%")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = WatermarkRemover(root)
    root.mainloop()
