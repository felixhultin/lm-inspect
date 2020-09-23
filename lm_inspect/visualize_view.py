from IPython.core.display import display, HTML, Javascript#, CSS
from string import Template

def visualize_inspector():
    html = HTML(filename='web/inspect.html')
    d3 = Javascript(filename='d3/d3.v4.min.js')
    js = Javascript(filename='web/visualize.js')
    display(html)
    display(d3)
    display(js)

def visualize(results, tokenizer):
    js_text = Javascript(filename='lm_inspect/web/visualize.js').data
    #css_text = CSS(filename='lm_inspect/web/inspect.css').data
    html_template = Template('''
        <body>
          <div id="container">
            <table id="model">
            </table>
            <table id="topk">
            </table>
          <div/>
        </body>
        <script>
            results = $results
            tokenizer = $tokenizer
            $js_text
        </script>
    ''')
    display(HTML(html_template.substitute({'js_text': js_text, 'results': results, 'tokenizer': tokenizer})))