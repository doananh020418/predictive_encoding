import PySimpleGUI as sg
import os.path
from DPCM import *
# First the window layout in 2 columns


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
        sg.FileBrowse(),
    ],
    [
        sg.Text("Image:"),
        sg.Image(key="-SHOW-"),
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Image reconstruct:")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FILE-":
        file = values["-FILE-"]
    #if event == "-TOUT2-":  # A file was chosen from the listbox
        try:
            filename = file
            print(filename)
            window["-SHOW-"].update(filename=filename)
            path = save_img(filename)
            window["-IMAGE-"].update(filename = path)

        except:
            pass

window.close()