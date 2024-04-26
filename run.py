#The terminal interfacer code
import os

print('''
##############################################################
#  Welcome to 3D motion capture made by Vaibhav Kumar Gupta  #
##############################################################

''')
while True:
    print('''
    # Enter Your Choice
    
    > image
    > camera
    > video
    > exit
    ''')
    choice = input("Enter your choice :")
    if (choice=="image"):
        image = input("Enter Full name of image :")
        os.system("python main.py --model trainedmodel.pth --image input/{}".format(image))
    elif (choice=="camera"):
        os.system("python main.py --model trainedmodel.pth --video 0")
    elif (choice=="video"):
        video = input("Enter Full name of Video :")
        os.system("python main.py --model trainedmodel.pth --video input/{}".format(video))
    elif (choice=="exit"):
        break
    else :
        print("Invalid choice !!!! Please enter appropriate choice")

