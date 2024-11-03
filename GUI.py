import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2 as cv
from copy import deepcopy
import esfCurveFitting_8
import gc
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)
import numpy as np


window_width = 1500
window_height = 1100
img_file_name = "current_frame.png"
picture = None
sample_coordinates = [0,0]
ROI_image = None
ROI_image_cropped = None
img1, img2 , img3 = None, None, None
img_box1, img_box2, img_box3 = None, None, None
desired_frequency = 0.2
desired_mtf = ""
alreadypicked = False
cap = None
img_box = None
mtf_box = None

# takes image width:x and height:y and scale x to max_width
def adapt_image_size(x, y, max_width):
    new_x = x * (max_width / x)
    new_y = y * (max_width / x)
    return (int(new_x),int(new_y))


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def findPointOnMaxEdge(img):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_x[sobel_x<0]=0 
    sobel_x = np.uint8(255*sobel_x/np.max(sobel_x)) 
    _ , sobel_thresh = cv.threshold(sobel_x, 200,255, cv.THRESH_BINARY) 

    nb_components, output, statss, centroids = cv.connectedComponentsWithStats(sobel_thresh, connectivity=8)
    sizes = statss[:, -1]  
    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]
    
    max_edge = np.zeros(output.shape)
    max_edge[output == max_label] = 255 
    x_coords,y_coords=np.where(max_edge==255)
    cx=round(np.mean(x_coords))
    cy=round(np.mean(y_coords))
    return cx,cy

def reset_picture(event, window):
    global sample_coordinates
    global ROI_image_cropped
    global alreadypicked
    
    sample_coordinates[0] = event.x
    sample_coordinates[1] = event.y
    window.config(cursor='')
    print(event.x, event.y)
    ROI_image_cropped = deepcopy(ROI_image[event.y-10:event.y+10, event.x-10:event.x+10])
    cv.imshow("ROI_OS",  ROI_image_cropped)
    cv.imwrite('./temp/ROI_OS.png', ROI_image_cropped)
    
    alreadypicked = True
    while True:
        if cv.waitKey(1) == ord('q'):
            break
        
        if cv.getWindowProperty('ROI_OS',cv.WND_PROP_VISIBLE) < 1:        
            break      

    cv.destroyAllWindows()
    
    
def display_image(window, frame):
    global img_file_name
    global picture
    global ROI_image

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # Frame is converted to RGB image.
    image = Image.fromarray(frame_rgb)
    
    # print(image.size[0])
    # print(image.size[1])
    x,y = adapt_image_size(image.size[0], image.size[1], 900)
    
    image_resized = image.resize((x, y))
    
    picture = ImageTk.PhotoImage(image_resized)
    
    temp_img = ImageTk.getimage(picture)
    # convert the PIL image to grayscale
    gray_img = temp_img.convert('L')
    # convert the grayscale PIL image to numpy array
    ROI_image = np.array(gray_img)
    #temp_img.save("./temp/temp.png", "png")
    # convert temp_img to gray-scale numpy
    #ROI_image = cv.imread('./temp/temp.png', 0)
    img_box.configure(image=picture)


def set_sample_point(window, event):
    window.config(cursor="cross")
    

def our_loop(window, event):
    global alreadypicked
    global sample_coordinates
    global cap

    cap = cv.VideoCapture(0) # 1 for External USB Camera, if it is 0 it takes the computer's webcam.
    fps = cap.get(cv.CAP_PROP_FPS)

    def update_frame():
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                #cv.imwrite('current_frame.png', frame)
                display_image(window, frame)
                if alreadypicked:
                    ROI_image_cropped = deepcopy(ROI_image[sample_coordinates[1]-10:sample_coordinates[1]+10, sample_coordinates[0]-10:sample_coordinates[0]+10])
                    # cv.imwrite('./temp/ROI_OS.png', ROI_image_cropped)#XXX
                    place_figures(window, ROI_image_cropped)
                    gc.collect()
        window.after(int(1000/fps), update_frame)  # Schedule the next frame update in 1 second--> düşürülebilir

    update_frame()  # Start the first frame update

    
def place_figures(window, ROI_image_cropped_):
    global desired_mtf
    global mtf_box
    mtf_matrix, esf_plt, lsf_plt, mtf_plt = esfCurveFitting_8.f1(ROI_image_cropped_)
    desired_mtf = mtf_matrix[np.abs(mtf_matrix[:, 1] - desired_frequency).argmin(),0]
    desired_mtf = round(desired_mtf, 3)
    mtf_text = f"MTF value = {desired_mtf} at frequency {desired_frequency}"
    print(mtf_text)
    mtf_box.configure(text=mtf_text)

    #time.sleep(0.025)
    
    global img1, img2, img3

    img1 = fig2img(esf_plt)
    img2 = fig2img(lsf_plt)
    img3 = fig2img(mtf_plt)
    gc.collect()

    """
    img1 = Image.open("./figures/EDGE_SPREAD_FUNCTION.png")
    img2 = Image.open("./figures/LINE_SPREAD_FUNCTION.png")
    img3 = Image.open("./figures/MODULATION_TRANSFER_FUNCTION.png")
    """
    img1 = ImageTk.PhotoImage(img1.resize((250,250)))
    img2 = ImageTk.PhotoImage(img2.resize((250,250)))
    img3 = ImageTk.PhotoImage(img3.resize((250,250)))
    # silinecek
    img_box1.configure(image=img1)
    img_box2.configure(image=img2)
    img_box3.configure(image=img3)
    gc.collect()
    
    print("Exiting function place_figures()")
    
    
def _quit(root):
    global cap  # Access the global camera capture object
    if cap is not None:
        cap.release()  # Release the camera capture object
    root.quit()
    root.destroy() 
    

def main():
    # create main window
    global alreadypicked
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root))
    root.geometry(f"{window_width}x{window_height}")
    # w,h = root.winfo_screenwidth(), root.winfo_screenheight()
    # root.geometry(f"{w}x{h}")
    root.minsize(1200, 900)
    root.title("ESF CURVE FITTING")


    # setup the grid
    root.columnconfigure(index=0, weight=1)
    root.columnconfigure(index=1, weight=8)
    root.columnconfigure(index=2, weight=4)
    # root.columnconfigure(index=3, weight=4)
    # root.rowconfigure(index=0, weight=4)
    # root.rowconfigure(index=1, weight=2)
    # root.grid_columnconfigure(index=0, minsize=300)
    # create widgets
    frame1 = ttk.Frame(root)
    frame1.grid(column=0, row=0, sticky=tk.NW, ipadx=10, ipady=10)

    global img_box
    frame2 = ttk.Frame(root)
    frame2.grid(column=1, row=0, sticky=tk.NW)
    img_box = ttk.Label(frame2)
    img_box.pack(side = tk.LEFT, expand= True)
    img_box.bind('<Button 1>', lambda e: reset_picture(event=e, window=root))

    global desired_mtf
    global mtf_box
    frame3 = ttk.Frame(root)
    frame3.grid(column=1, row=1, sticky=tk.NW)
    mtf_box = ttk.Label(frame3, text=desired_mtf)
    mtf_box.pack(side=tk.LEFT, expand=True)

    global img_box1, img_box2, img_box3
    frame3 = ttk.Frame(root)
    frame3.grid(column=2, row=0, sticky=tk.N)
    img_box1 = ttk.Label(frame3, padding= 10)
    img_box2 = ttk.Label(frame3, padding= 10)
    img_box3 = ttk.Label(frame3, padding= 10)
    img_box1.pack()
    img_box2.pack()
    img_box3.pack()

    pick_btn = ttk.Button(frame1, text="Pick Sample")
    pick_btn.pack(padx=5, pady=5, fill='x')
    pick_btn.bind('<Button 1>', lambda e: set_sample_point(root, event=e))
    
    start_btn = ttk.Button(frame1, text="start button")
    start_btn.pack(padx=5, pady=5, fill='x')
    start_btn.bind('<Button 1>', lambda e: our_loop(root, event=e))
    
   
    # ttk.Separator(root,orient=tk.VERTICAL).grid(column=0, row=0, rowspan=2, sticky=tk.NS)
        

    root.mainloop()
    root.quit()
    
    
main()
        
