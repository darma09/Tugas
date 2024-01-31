import turtle
import streamlit as st

def bintang():
    s = turtle.Screen()
    t = turtle.Turtle()
    s.bgcolor('black')
    t.pencolor('white')
    t.speed(2)
    t.lt(100)
    t.fd(100)
    col = ('blue', 'red', 'white', 'green')

    for n in range(4):
        t.penup()
        t.goto(0, -n * 40)
        t.pendown()
        t.pencolor(col[n % 4])

        for _ in range(5):
            t.forward(100)
            t.right(144)
            
    s.exitonclick()

def main():
    st.title("Bintang Animation")
    st.write("Click the button to start the animation")
    if st.button("Start Animation"):
        bintang()

if __name__ == '__main__':
    main()
