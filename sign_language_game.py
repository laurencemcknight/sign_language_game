import cv2
from tkinter import *
import time
import random
from CNN import Cnn
from webcam_image_processor import Webcam_processor
import os


class Game():
    def __init__(self, Webcam, Cnn): 
        """
        Set up game screen GUI    
        Paramaters:
         Webcam: the Webcam class that will be used to capture images of the user's sign language gestures
         Cnn: the CNN class that will be used for classifying the images captured by the webcam
        Returns:
         None
        """
        self.window = Tk()
        self.window.geometry("710x120")    
        self.window.configure(bg="white")

        self.instructions_label = Label(self.window,text="Please translate the following words:",font='vardana 27',fg='black', bg="white")
        self.feedback_label = Label(self.window,font='vardana 20',fg='black', bg="white")
        self.sentince_label = Text(self.window,font='vardana 10',fg='black', bg="white") 
        self.sentince_label.insert(INSERT, ', '.join(sign_language_sentince))

        self.instructions_label.grid(row=1,column=1)
        self.feedback_label.grid(row=2,column=1)
        self.sentince_label.grid(row=3,column=1)

        self.window.update()

        print("Press q to quit game.\n")
        frames_num_list = Webcam.game_screen(Cnn, sign_language_sentince, self.window, self.sentince_label, self.feedback_label, compliments)
        Webcam.camera.release()
        cv2.destroyAllWindows()  
        if frames_num_list:
            self.setup_end_game_screen(frames_num_list)

    
    def setup_end_game_screen(self, frames_num_list):
        """
        Set up the end game screen which will give the user feedback on the words they translated well and poorly
        Paramaters:
         frames_num_list: a list of the number of frames taken for the user to correctly gesture each word
        Returns:
         None
        """
        self.instructions_label.destroy()
        self.feedback_label.destroy()
        self.sentince_label.destroy()
        self.window.update()
        self.window.destroy()

        self.window = Tk()
        self.window.geometry("600x90")  

        self.window.configure(bg="white")

        compliment = compliments[random.randint(0, len(compliments)-1)]
        final_label = Label(self.window,text=compliment + " Translation Completed!",font='vardana 20',fg='black', bg="white")
        best_word = sign_language_sentince[frames_num_list.index(min(frames_num_list))]
        best_word_label = Label(self.window,text="Good job translating '" + best_word + "'. You translated '" + best_word + "' the fastest of all words!",font='vardana 10',fg='black', bg="white")
        worst_word = sign_language_sentince[frames_num_list.index(max(frames_num_list))]
        worst_word_label = Label(self.window,text="You took the longest when translating '" + worst_word + "'. Try practicing with '" + worst_word + "' more.",font='vardana 10',fg='black', bg="white")

        final_label.grid(row=1,column=1)
        best_word_label.grid(row=2,column=1)
        worst_word_label.grid(row=3,column=1)

        self.window.update()
        time.sleep(5)
        self.window.destroy()


sign_language_words = ["Why", "Bus", "Yes", "Fine", "Hello", "Stop"] #feel free to add more words to the list to practice 
num_of_words_to_play = 2
sign_language_sentince = random.sample(sign_language_words, num_of_words_to_play)
compliments = ["Nice!", "Good Job!", "Fantastic!", "Excellent!", "All Right!"]

Cnn = Cnn(sign_language_words)
Webcam = Webcam_processor()

dir_items = os.listdir(os.path.dirname(os.path.realpath(__file__)))
if "data" not in dir_items or "checkpoint.pth" not in dir_items:
    #if -s is given as a argument to the program then run the program setup. This is done by creating training images and then training the CNN on those images.
    Webcam.create_training_images(sign_language_words)
    print("Training images created.\n")

    epochs = 7
    Cnn.iter_epoch(epochs)
    print("CNN successfully trained")

    print("Setup complete\n")

Game(Webcam, Cnn)