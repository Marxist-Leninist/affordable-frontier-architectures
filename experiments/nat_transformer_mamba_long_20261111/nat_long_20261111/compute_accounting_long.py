"""Static, partial operation accounting only; no model/data imports or execution."""
import json
from pathlib import Path

def accounting():
    batch, length, width, layers, output, vocab = 1,128,128,3,32,256
    transformer=layers*(length*(4*width**2+2*width*654)+2*length**2*width)+output*width*vocab
    mamba=layers*2*(length*(width*512+256*40+8*256+256*width)+(length+3)*256*4)+output*width*vocab
    states=layers*2*length*256*16
    return {
        'scope':'Static partial forward-pass operation estimate, not a measurement or complete FLOP count.',
        'shape':{'batch':batch,'sequence_positions':length,'width':width,'layers':layers,'output_positions':output,'byte_vocab':vocab},
        'transformer_matrix_macs':transformer,
        'transformer_formula':'3*(128*(4*128^2+2*128*654)+2*128^2*128)+32*128*256',
        'mamba_matrix_and_convolution_macs':mamba,
        'mamba_formula':'3*2*(128*(128*512+256*40+8*256+256*128)+131*256*4)+32*128*256',
        'convolution_accounting':'Kernel4, padding3 produces131 positions before truncation to128; all131 nominal convolution outputs are counted.',
        'extra_convolution_macs_for_discarded_positions':layers*2*3*256*4,
        'mamba_state_elements_updated_per_forward':states,
        'state_elements_formula':'3*2*128*256*16',
        'excluded':['normalization','activations','RoPE','softmax','selective scan recurrence','other pointwise operations'],
        'caveats':[
            'MAC counts cover listed matrix/convolution operations only. One MAC is one multiply-accumulate, not a complete FLOP accounting.',
            'Mamba state-element updates are a separate count, not MACs or FLOPs, and must not be added to the partial MAC totals.',
            'No training backward, optimizer, memory movement, dispatch or compilation accounting is included.',
            'Matching parameters is not matching compute. These estimates do not establish equal FLOPs or less total work for Mamba.',
            'Both dense arms invoke all layer modules. Embeddings use indexed rows; neither arm has an expert router.',
            'Interpret partial accounting alongside measured whole-model runtime, not as a substitute for timing.'
        ]
    }

def main():
    x=accounting()
    assert x['transformer_matrix_macs']==103088128
    assert x['mamba_matrix_and_convolution_macs']==86788096
    assert x['extra_convolution_macs_for_discarded_positions']==18432
    assert x['mamba_state_elements_updated_per_forward']==3145728
    assert json.loads(json.dumps(x))==x
    path=Path(__file__).with_name('compute_accounting_long.json')
    path.write_text(json.dumps(x,indent=2)+'\n')
    print('PASS: stdlib static formulas and JSON round-trip; no model execution.')

if __name__=='__main__': main()
