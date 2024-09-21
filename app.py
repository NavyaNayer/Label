import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import torch

class ImageLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Labeler")

        self.frame = tk.Frame(master)
        self.frame.pack()

        self.canvas = tk.Canvas(self.frame, bg='white', width=800, height=600)
        self.canvas.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.add_bbox_button = tk.Button(self.button_frame, text="Add Bounding Box", command=self.add_bbox)
        self.add_bbox_button.pack(pady=10)

        self.image_id = None
        self.bboxes = []
        self.selected_bbox = None
        self.start_x = None
        self.start_y = None
        self.is_drawing = False
        self.is_resizing = False
        self.dragging = False
        self.current_label = "Unknown"
        self.current_bbox = None

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.image = self.original_image.copy()
            self.display_image(self.image)
            self.perform_detection()

    def perform_detection(self):
        results = self.model(self.image)
        detections = results.xyxy[0].numpy()  # Get predictions
        self.bboxes = []
        self.canvas.delete("all")
        self.display_image(self.image)

        for *xyxy, conf, cls in detections:
            bbox_id = self.draw_bbox(xyxy, results.names[int(cls)])
            self.bboxes.append(bbox_id)  # Store the ID of the bbox

    def draw_bbox(self, bbox, label):
        x1, y1, x2, y2 = map(int, bbox)
        bbox_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        self.canvas.create_text(x1, y1, text=label, fill='white', font=('Arial', 12, 'bold'))
        return bbox_id

    def display_image(self, image):
        max_size = (800, 600)
        image.thumbnail(max_size, Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_click(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.selected_bbox = self.find_bbox(event.x, event.y)

        if self.selected_bbox:
            coords = self.canvas.coords(self.selected_bbox)
            if self.is_near_corner(event.x, event.y, coords):
                self.is_resizing = True  # Start resizing
            else:
                self.dragging = True  # Start dragging
        else:
            self.is_drawing = True
            self.current_label = simpledialog.askstring("Input", "Enter label for new object:")
            if self.current_label:
                self.current_bbox = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red', width=2)
            else:
                self.is_drawing = False

    def is_near_corner(self, x, y, coords):
        corner_distance = 10  # Pixels to define a corner
        return ((abs(x - coords[0]) < corner_distance and abs(y - coords[1]) < corner_distance) or
                (abs(x - coords[2]) < corner_distance and abs(y - coords[3]) < corner_distance))

    def find_bbox(self, x, y):
        for bbox_id in self.bboxes:
            coords = self.canvas.coords(bbox_id)
            if len(coords) == 4:  # Ensure coords has 4 elements
                if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                    return bbox_id
        return None

    def on_drag(self, event):
        if self.is_drawing and self.current_bbox:
            # Resize the bounding box while dragging
            self.canvas.coords(self.current_bbox, self.start_x, self.start_y, event.x, event.y)
        elif self.selected_bbox:
            coords = self.canvas.coords(self.selected_bbox)
            if self.is_resizing:
                # Resize the bounding box
                new_x2 = max(event.x, coords[0] + 1)  # Keep at least 1 pixel width
                new_y2 = max(event.y, coords[1] + 1)  # Keep at least 1 pixel height
                self.canvas.coords(self.selected_bbox, coords[0], coords[1], new_x2, new_y2)
            elif self.dragging:
                # Drag the bounding box
                dx = event.x - self.start_x
                dy = event.y - self.start_y
                new_coords = [coords[0] + dx, coords[1] + dy, coords[2] + dx, coords[3] + dy]
                self.canvas.coords(self.selected_bbox, new_coords)
                self.start_x, self.start_y = event.x, event.y

    def on_release(self, event):
        if self.is_drawing and self.current_bbox:
            # Finalize the bounding box after drawing
            if self.start_x != event.x or self.start_y != event.y:
                self.bboxes.append(self.current_bbox)
            else:
                self.canvas.delete(self.current_bbox)  # Remove if no area was drawn
            self.current_bbox = None
            
        self.is_drawing = False
        self.dragging = False
        self.is_resizing = False
        self.selected_bbox = None

    def add_bbox(self):
        self.current_label = simpledialog.askstring("Input", "Enter label for new object:")
        if self.current_label:
            messagebox.showinfo("Info", "Click on the canvas to place the bounding box.")
        else:
            messagebox.showwarning("Warning", "No label provided.")

    def run(self):
        self.master.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    app.run()