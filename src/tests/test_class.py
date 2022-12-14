from inference import input_fn, model_fn, output_fn, predict_fn

class TestClass:
    '''The class contains multiple functions to test
    the defined by Crystal Gazers predictor model.'''

    model = model_fn('.')
    accepted_output_test1 = ['enamorat', 'cansat', 'embarassat', 'fugint']
    accepted_output_test2 = ['enamorat', 'cansat', 'embarassat', 'fugint']
    accepted_output_test3 = ['voleibol', 'futbol', 'tenis', 'equip']


    def test_one(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        input_one = input_fn('{"input": ["el", "noi", "està", "de", "sa", "nòvia"]}',
                             'application/json')
        pred_one = predict_fn(input_one, self.model)
        out_one = output_fn(pred_one, 'application/json')[1:-1]
        assert out_one in self.accepted_output_test1

    def test_two(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''
        input_two = input_fn('{"input": ["el", "noi", "està", "del", "seu", "nòvio"]}',
                             'application/json')
        pred_two = predict_fn(input_two, self.model)
        out_two = output_fn(pred_two, 'application/json')[1:-1]
        assert out_two in self.accepted_output_test2
    def test_three(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''
        input_three = input_fn('{"input": ["la", "jugadora", "de", "va", "guanyar", "molt"]}',
                              'application/json')
        pred_three = predict_fn(input_three, self.model)
        out_three = output_fn(pred_three, 'application/json')[1:-1]
        assert out_three in self.accepted_output_test3
