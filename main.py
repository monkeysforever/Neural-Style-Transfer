from argparse import ArgumentParser
from PIL import Image
from style_transfer import run_style_transfer

ITERATIONS = 1000
CONTENT_PATH = "C:/Users/ahlaw/Desktop/content.jpg" 
STYLE_PATH = "C:/Users/ahlaw/Desktop/style.jpg" 
OUTPUT_PATH = "output.jpg"
PRINT_ITERATIONS = 100
ALPHA = 1e3
BETA = 1e-2
LEARNING_RATE = 5
BETA1 = 0.99
BETA2 = 0.999
EPSILON = 1e-1
POOLING = 'avg'
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content-path',
            dest='content_path', help='content image path',
            default = CONTENT_PATH)             
    parser.add_argument('--style-path',
            dest='style_path',
            help='style image path', default = STYLE_PATH)
    parser.add_argument('--content-layers', type = str,
            nargs = '*', dest='content_layers', help='content layers',
            default = CONTENT_LAYERS)
    parser.add_argument('--style-layers', type = str,
            nargs = '*', dest='style_layers', help='style layers',
            default = STYLE_LAYERS)
    parser.add_argument('--output-path',
            dest='output_path', help='output path',
            default = OUTPUT_PATH)
    parser.add_argument('--iterations', type=int,
            dest='iterations', help='iterations (default %(default)s)',
            default=ITERATIONS)
    parser.add_argument('--print-iterations', type=int,
            dest='print_iterations', help='statistics printing frequency',
            default = PRINT_ITERATIONS)    
    parser.add_argument('--alpha', type=float,
            dest='alpha', help='content weight (default %(default)s)',
            default=ALPHA)
    parser.add_argument('--beta', type=float,
            dest='beta', help='style weight (default %(default)s)',
            default=BETA)    
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            default=EPSILON)        
    parser.add_argument('--pooling',
            dest='pooling',
            help='pooling layer configuration: max or avg (default %(default)s)',
            default=POOLING)
    
    return parser

def main():
    parser = build_parser()
    options = parser.parse_args()
    config = {'content_path' : options.content_path,
              'style_path' : options.style_path,
              'style_layers': options.style_layers,
              'content_layers': options.content_layers,             
              'iterations': options.iterations,
              'print_iterations': options.print_iterations,
              'alpha': options.alpha,
              'beta': options.beta,
              'beta1': options.beta1,
              'beta2': options.beta2,
              'epsilon': options.epsilon,
              'pooling': options.pooling,
              'learning_rate': options.learning_rate}
    
    best, loss = run_style_transfer(**config)         
    Image.fromarray(best).save(options.output_path)
    
if __name__ == '__main__':
    main()