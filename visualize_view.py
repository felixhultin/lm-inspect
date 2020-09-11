#from .lm_inspect import Inspector
from IPython.core.display import display, HTML, Javascript

def visualize_inspector():
    html = HTML(filename='web/inspect.html')
    d3 = Javascript(filename='d3/d3.v4.min.js')
    js = Javascript(filename='web/vizualize.js')
    display(html)
    display(d3)
    display(js)
