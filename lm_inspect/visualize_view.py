from IPython.core.display import display, HTML, Javascript
from string import Template

def visualize_inspector():
    html = HTML(filename='web/inspect.html')
    d3 = Javascript(filename='d3/d3.v4.min.js')
    js = Javascript(filename='web/visualize.js')
    display(html)
    display(d3)
    display(js)

def visualize():
    js_text = Javascript(filename='web/visualize.js').data
    html_template = Template('''
        <body>
          <div id="container">
            <table id="model">
            </table>
            <table id="topk">
            </table>
          <div/>
        </body>
        <script>$js_text</script>
    ''')
    #return html_template
    display(HTML(html_template.substitute({'js_text': js_text})))
