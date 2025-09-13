import RPi.GPIO as GPIO
from time import sleep


GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)


pwm=GPIO.PWM(7, 50)
pwm.start(0)


def aimAngle(angle): # danh cho quay banh rang lon ( 0 -> 360)
    global current_angle
    duty = angle/40 + 2
    GPIO.output(7, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(7, False)
    pwm.ChangeDutyCycle(duty)
        
    current_angle = angle

while True:
    aimAngle(360)
    
    
    
    