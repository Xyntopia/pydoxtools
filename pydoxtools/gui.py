import justpy as jp

def hello_world2():
    wp = jp.WebPage()
    for i in range(1,11):
        jp.P(text=f"{i}) Hello World!", a=wp, style=f"font-size: {10*i}px")
    return wp

jp.justpy(hello_world2)
