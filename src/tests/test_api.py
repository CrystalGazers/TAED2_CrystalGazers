from inference import fetch, model_fn


class TestClass:
    '''The class contains multiple functions to test
    the defined by Crystal Gazers API.'''

    model = model_fn('.')
    accepted_output_test1 = ['enamorat', 'cansat', 'embarassat', 'fugint']
    accepted_output_test2 = ['enamorat', 'cansat', 'embarassat', 'fugint']
    accepted_output_test3 = ['voleibol', 'futbol', 'tenis', 'equip']
    def test_one(self):
        '''The function checks whether 2the given input is
        contained in the corresponding accepted output'''

        out_one, status_code = fetch(["el", "noi", "està", "de", "sa", "nòvia"])
        assert out_one in self.accepted_output_test1
        assert status_code == 200

    def test_two(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        out_two, status_code = fetch(["el", "noi", "està", "del", "seu", "nòvio"])
        assert out_two in self.accepted_output_test2
        assert status_code == 200

    def test_three(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        out_three, status_code = fetch(["la", "jugadora", "de", "va", "guanyar", "molt"])
        assert out_three in self.accepted_output_test3
        assert status_code == 200
