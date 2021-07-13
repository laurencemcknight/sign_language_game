import os
import torch
from torchvision import transforms, models, datasets
import matplotlib.pyplot as pt
import cv2


class Cnn():
    def __init__(self,sign_language_words):
        """
        Initialize the CNN 
        Paramaters:
         sign_language_words: a list of all the sign language words that the CNN will classify
        Returns:
         None
        """
        self.sign_language_words = sign_language_words
        torch.cuda.empty_cache()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.img_transformer = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        self.model = models.vgg16(pretrained=True)
        for parameter in self.model.features.parameters():
            parameter.requires_grad = False 
        output_layer_inputs = self.model.classifier[6].in_features
        desired_output_layer_outputs = len(self.sign_language_words)
        output_layer = torch.nn.Linear(output_layer_inputs, desired_output_layer_outputs)
        self.model.classifier[6] = output_layer #transfer learning with the last layer of the model
        self.model.to(self.device)
        self.criterion_function = torch.nn.CrossEntropyLoss()
        self.optimizer_function = torch.optim.Adam(self.model.parameters(), lr = 0.0001)
        self.checkpoint_location = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint.pth")
        

    def iter_epoch(self, epochs):
        """
        Itterate the CNN though training and testing epochs
        Paramaters:
         epochs: the number of epochs to itterate through
        Returns:
         None
        """
        training_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "training"), transform=self.img_transformer) 
        training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=20, shuffle=True)
        testing_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "testing"), transform=self.img_transformer)
        testing_loader = torch.utils.data.DataLoader(testing_dataset, batch_size = 20, shuffle=False) 

        self.training_losses = []
        self.training_accuracy = []
        self.testing_losses = []
        self.testing_accuracy = []
        
        for itteration in range(epochs):
            print("Processing Epoch: " + str(itteration+1) + "...")
            self.train(training_loader)
            self.test(testing_loader)
            
        checkpoint = {'model': self.model,
                      'state_dict': self.model.state_dict(),
                      'optimizer' : self.optimizer_function.state_dict()}
        torch.save(checkpoint, self.checkpoint_location)
        self.data()
        
    
    def train(self, training_loader):
        """
        Train the CNN
        Paramaters:
         training_loader: the data loader for the images the CNN will be trained on 
        Returns:
         None
        """
        current_loss = 0.0
        current_accuracy = 0.0
        for images, labels in training_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            model_outputs = self.model(images)
            model_outputs = model_outputs.to(self.device)
            loss = self.criterion_function(model_outputs, labels)
            self.optimizer_function.zero_grad()
            loss.backward()
            self.optimizer_function.step()
            current_loss += loss.item()
            _, prediction = torch.max(model_outputs, 1)
            current_accuracy += torch.sum(prediction == labels.data) 
        epoch_loss = current_loss/len(training_loader.dataset)
        self.training_losses.append(epoch_loss)
        epoch_accuracy = (current_accuracy/len(training_loader.dataset)).cpu().numpy()
        self.training_accuracy.append(epoch_accuracy * 100) 


    def test(self, testing_loader): 
        """
        Test the CNN's accuracy after being trained
        Paramaters:
         training_loader: the data loader for the images the CNN will be trained on 
        Returns:
         None
        """
        current_loss = 0.0
        current_accuracy = 0.0        
        self.confusion_matrix = torch.zeros(len(self.sign_language_words), len(self.sign_language_words))        
        for images, labels in testing_loader: 
            images, labels = images.to(self.device), labels.to(self.device)
            model_outputs = self.model(images)
            loss = self.criterion_function(model_outputs, labels)        
            current_loss += loss.item() 
            _, prediction = torch.max(model_outputs, 1)
            current_accuracy += torch.sum(prediction == labels.data)      
            for t, p in zip(labels.view(-1), prediction.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1           
        epoch_loss = current_loss/len(testing_loader.dataset)
        self.testing_losses.append(epoch_loss)
        epoch_accuracy = (current_accuracy/len(testing_loader.dataset)).cpu().numpy()
        self.testing_accuracy.append(epoch_accuracy * 100)    
        

    def data(self):
        """
        Graph the change in CNN accuracy over the epochs and print the testing accuracy of each word 
        Paramaters:
         None 
        Returns:
         None
        """     
        pt.figure()
        pt.plot(self.training_losses, label='training loss')
        pt.plot(self.testing_losses, label='testing loss')
        pt.ylabel("Loss (Scale 0 - 1)")
        pt.xlabel("Epoch")
        pt.legend()        
        
        pt.figure()
        pt.plot(self.training_accuracy, label='training accuracy')
        pt.plot(self.testing_accuracy, label='testing accuracy')
        pt.ylabel("Accuracy %")
        pt.xlabel("Epoch")
        pt.legend()        
        pt.show()
        
        accuracy_array = self.confusion_matrix.diag()/self.confusion_matrix.sum(1)
        self.testing_class_accuracy = {}        
        for i in range(len(self.sign_language_words)):
            self.testing_class_accuracy[self.sign_language_words[i]] = str(round((accuracy_array.data[i].item() * 100), 1)) + "%"  

        print("Individual word testing accuracies:")          
        for word in self.testing_class_accuracy.keys():
            print("'" + word + "' accuracy: " + str(self.testing_class_accuracy[word]))
    

    def init_for_classify_image(self):
        """
        Set up image classification by loading the trained CNN  
        Paramaters:
         None 
        Returns:
         None
        """
        checkpoint = torch.load(self.checkpoint_location)
        trained_model = checkpoint['model']
        trained_model.load_state_dict(checkpoint['state_dict'])
        for parameter in trained_model.parameters():
            parameter.requires_grad = False
        trained_model.eval()   
        self.trained_model = trained_model.to(self.device)        
        

    def classify_image(self, image):
        """
        Classify a image of the user performing a sign language gesture as one of the words in the sign_language_words list
        Paramaters:
         image: a image frame captured by the webcam to be classified 
        Returns:
         predicted_word: The predicted sign language word 
        """     
        image = cv2.resize(image, (224, 224))
        image = transforms.ToTensor()(image).unsqueeze(0)
        image = image.to(self.device)
        model_output = self.trained_model(image)
        _, prediction = torch.max(model_output, 1)
        predicted_word = self.sign_language_words[prediction]
        return predicted_word