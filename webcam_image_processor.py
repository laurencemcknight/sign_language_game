import cv2
import os
from pathlib import Path
import time
import random


class Webcam_processor():  
    def image_preprocessing(self, segmented_hand, frame, frame_copy):
        """
        Using image preprocessing to remove the background (ie: everything but the users hand) from the image
        Paramaters:
         segmented_hand: a tuple of the thresholded hand image and a image of the segmented hand
         frame: a frame captured by the user's webcam
         frame_copy: a copy of frame (used for displaying an overlay of the users segmented hand)
        Returns:
         just_hand_image: a modified version of frame with everything but the users hand removed from the image 
        """
        hand_mask, segmented = segmented_hand
        cv2.drawContours(frame_copy, [segmented + (self.hand_area_right, self.hand_area_top)], -1, (0, 0, 255))
        hand_image = frame.copy()
        hand_image = hand_image[self.hand_area_top : self.hand_area_bottom, self.hand_area_right : self.hand_area_left]
        just_hand_image = cv2.bitwise_and(hand_image, hand_image, mask=hand_mask)
        return just_hand_image


    def create_blured_grayscale_image(self, frame):
        """
        Create blured greyscaled version of the current webcam frame
        Paramaters:
         frame: a frame captured by the user's webcam
        Returns:
         blured_grayscale_image: blured greyscaled webcam frame
        """
        hand_area = frame[self.hand_area_top : self.hand_area_bottom, self.hand_area_right : self.hand_area_left]
        grayscale_image = cv2.cvtColor(hand_area, cv2.COLOR_BGR2GRAY)
        blured_grayscale_image = cv2.GaussianBlur(grayscale_image, (7, 7), 0)
        return blured_grayscale_image


    def background_avg(self, grayscale_image):
        """
        Create a average of the image background to use in image subtraction 
        Paramaters:
         grayscale_image: a greyscaled version of a frame captured by the user's webcam
        Returns:
         None
        """
        background = grayscale_image.copy()
        if self.background is None:
            self.background = background.astype("float")
            return
        cv2.accumulateWeighted(grayscale_image, self.background, 0.5)
        

    def hand_segmentation(self, grayscale_image):
        """
        Segment the user's hand using thresholding  
        Paramaters:
         grayscale_image: a greyscaled version of a frame captured by the user's webcam
        Returns:
         thresholded_img: a thresholded image of grayscale_image
         segmented_hand: a image of the users segmented hand
        """
        difference = cv2.absdiff(self.background.astype("uint8"), grayscale_image)
        thresholded_img = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1] 
        contours, _ = cv2.findContours(thresholded_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        else:
            segmented_hand = max(contours, key=cv2.contourArea)
            return (thresholded_img, segmented_hand)


    def initialize_camera_vars(self):
        """
        Initalize variables that will be used in capturing webcamera images  
        Paramaters:
         None
        Returns:
         None
        """
        self.background = None
        self.camera = cv2.VideoCapture(0)
        self.hand_area_top, self.hand_area_bottom = 10, 225 
        self.hand_area_right, self.hand_area_left = 350, 590
        self.overall_itteration_num = 0
        self.min_frame_itterations = 30


    def update_game_GUI(self, current_word_index, word, window, sentince_label, feedback_label, compliments):
        """
        Update the games GUI when the user correctly gestures a word by turing the word green and giving the user a compliment
        Paramaters:
         current_word_index: the index of the first letter of word in sign_language_sentince
         word: the word that the user has correctly gesutred
         window: the window of the game screen GUI
         sentince_label: a label in the game screen GUI used to display the words in sign_language_sentince
         feedback_label: a label in the game screen GUI used for giving the user positive encouragment when they get a word correct
         compliments: a list of compliment strings
        Returns:
         None
        """
        current_word_index += len(word) + 2
        sentince_label.tag_add("start", "1.0", "1."+str(current_word_index))
        sentince_label.tag_config("start", background="green") 
        compliment = compliments[random.randint(0, len(compliments)-1)]
        feedback_label.config(text=compliment + " You correctly translated " + word)
        window.update()


    def make_image_dirs(self, word):
        """
        Create directories to store images of a word being gestured in sign language (to be used to train and test the CNN)  
        Paramaters:
         word: the word that will be gestured in sign language
        Returns:
         full_training_dir: directory pathway to training directory for images of the word being gestured in sign language
         full_testing_dir: directory pathway to testing directory for images of the word being gestured in sign language
        """
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        training_dir = "training"
        testing_dir = "testing"
        full_training_dir = os.path.join(os.path.join(data_dir, training_dir), word)
        full_testing_dir = os.path.join(os.path.join(data_dir, testing_dir), word)
        Path(full_training_dir).mkdir(parents=True, exist_ok=True)
        Path(full_testing_dir).mkdir(parents=True, exist_ok=True)
        return (full_training_dir, full_testing_dir)

                
    def game_screen(self, Cnn, sign_language_sentince, window, sentince_label, feedback_label, compliments):
        """
        Set up webcam screen GUI to classify preprocessed frames of the users sign language gestures
        Paramaters:
         Cnn: the CNN class that will be used for classifying the images captured by the webcam
         sign_language_sentince: a list of words for the user to gesure in sign language
         window: the window of the game screen GUI
         sentince_label: a label in the game screen GUI used to display the words in sign_language_sentince
         feedback_label: a label in the game screen GUI used for giving the user positive encouragment when they get a word correct
         compliments: a list of compliment strings
        Returns:
         frames_num_list: a list of the number of frames taken for the user to correctly gesture each word
        """
        current_word_index = 0
        frames_num_list = []
        Cnn.init_for_classify_image()
        self.initialize_camera_vars()
        for word in sign_language_sentince:
            itteration_num = 0
            correct_translation = False
            while correct_translation is False:
                _, frame = self.camera.read()
                frame = cv2.flip(frame, 1)
                frame_copy = frame.copy()
                blured_grayscale_image = self.create_blured_grayscale_image(frame)
                self.overall_itteration_num += 1
                if self.overall_itteration_num <= self.min_frame_itterations: 
                    #use the first 30 frames to get a averaged background image for image subtraction
                    self.background_avg(blured_grayscale_image)
                else:
                    segmented_hand = self.hand_segmentation(blured_grayscale_image)
                    if segmented_hand is not None:
                        itteration_num += 1
                        just_hand_image = self.image_preprocessing(segmented_hand, frame, frame_copy)
                        predicted_word = Cnn.classify_image(just_hand_image)
                        cv2.putText(frame_copy, "Translate: " + str(word), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,1,0), 2)
                        cv2.putText(frame_copy, "You are signing: " + str(predicted_word), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,1,0), 2)
                        if predicted_word == word:
                            correct_translation = True
                            self.update_game_GUI(current_word_index, word, window, sentince_label, feedback_label, compliments)
                            frames_num_list.append(itteration_num) 
                cv2.rectangle(frame_copy, (self.hand_area_left, self.hand_area_top), (self.hand_area_right, self.hand_area_bottom), (0,255,0), 2)
                cv2.imshow("Translation Game", frame_copy)
                key_press = cv2.waitKey(1) & 0xFF
                if key_press == ord("q"):
                    #quit the game if q is pressed
                    self.camera.release()
                    cv2.destroyAllWindows()   
                    return
        return frames_num_list
        

    def create_training_images(self, sign_language_words): 
        """
        Create training images using image preprocessing  
        Paramaters:
         sign_language_words: a list of all the sign language words that the user will need to sign
        Returns:
         None
        """
        self.initialize_camera_vars()
        for word in sign_language_words:
            image_num = 0
            stay_on_current_word= True
            full_training_dir, full_testing_dir = self.make_image_dirs(word)
            while(stay_on_current_word):
                _, frame = self.camera.read()
                frame = cv2.flip(frame, 1)
                frame_copy = frame.copy()
                blured_grayscale_image = self.create_blured_grayscale_image(frame)
                cv2.putText(frame_copy, "Sign: " + str(word) +" in the red box", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,1,0), 2)
                cv2.putText(frame_copy, "Press 'n' for next word", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,1,0), 2)
                self.overall_itteration_num += 1
                if self.overall_itteration_num <= self.min_frame_itterations:
                    self.background_avg(blured_grayscale_image)
                else:
                    segmented_hand = self.hand_segmentation(blured_grayscale_image)
                    if segmented_hand is not None:
                        just_hand_image = self.image_preprocessing(segmented_hand, frame, frame_copy)
                        if image_num%5 == 0: #one in 5 images will be for testing 
                            train_or_test_dir = full_testing_dir
                        else:
                            train_or_test_dir =  full_training_dir
                        save_as = os.path.join(train_or_test_dir, word + str(image_num) + '.jpg')
                        cv2.imwrite(save_as, just_hand_image)
                cv2.rectangle(frame_copy, (self.hand_area_left, self.hand_area_top), (self.hand_area_right, self.hand_area_bottom), (0,255,0), 2)
                cv2.imshow("Translation Game", frame_copy)
                image_num += 1
                key_press = cv2.waitKey(1) & 0xFF
                if key_press == ord("n"): #to move to the next word
                    if word == sign_language_words[-1]:
                        self.camera.release()
                        cv2.destroyAllWindows()   
                        return
                    else:
                        stay_on_current_word = False