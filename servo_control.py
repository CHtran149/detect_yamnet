#Example Servo Code
#Control the angle of a 
#Servo Motor with Raspberry Pi

# free for use without warranty
# www.learnrobotics.org
import RPi.GPIO as GPIO
from time import sleep
from multiprocessing.connection import Listener


address = ('localhost', 6000)

listener = Listener(address, authkey=b'thanhcolonhue')
print('ready to accept connection')

conn = listener.accept()
print('connection accepted from', listener.last_accepted)
print('Server started, please run the file thuchanh.py')


GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)


GPIO.setup(13, GPIO.OUT)

def onOffLaser():
    GPIO.output(13, GPIO.HIGH)
    

pwm=GPIO.PWM(7, 50)
pwm.start(0)

def setAngle(angle):# danh cho quay banh xang vua dong co ( quay tu 0 -> 180)
    duty = angle / 18 + 2
    GPIO.output(7, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(7, False)
    pwm.ChangeDutyCycle(duty)


def aimAngle(angle): # danh cho quay banh rang lon ( 0 -> 360)
    global current_angle
    duty = angle/40 + 2
    GPIO.output(7, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(7, False)
    pwm.ChangeDutyCycle(duty)
        
    current_angle = angle
    
    

onOffLaser() # Bat Laser
aimAngle(0)  # dat ve goc 0 do


while True:
    
    msg = conn.recv()  # Nhan message
    angle = int(msg) 
    
    print("Angle: " + msg)
    aimAngle(angle)

    sleep(1)

pwm.stop()
GPIO.cleanup()
listener.close()
