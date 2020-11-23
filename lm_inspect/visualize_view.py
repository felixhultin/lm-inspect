import json

from IPython.core.display import display, HTML, Javascript
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
    css_text = open('lm_inspect/web/inspect.css').read()
    html_template = Template('''
        <body>
          <div id="container">
            <table id="model">
            </table>
            <table id="topk">
            </table>
            <table id="last">
            </table>
          <div/>
        </body>
        <style>$css_text</style>
        <script>
            results = $results
            tokenizer = $tokenizer
            $js_text
        </script>
    ''')
    template_vars = {'js_text': js_text, 'results': json.dumps(results), 'tokenizer': tokenizer, 'css_text': css_text}
    display(HTML(html_template.substitute(template_vars)))
