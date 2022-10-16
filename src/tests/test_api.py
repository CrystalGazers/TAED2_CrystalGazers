from inference import fetch, model_fn


class TestClass:
    '''The class contains multiple functions to test
    the defined by Crystal Gazers API.'''

    model = model_fn('.')
    accepted_output_test1 = ['va', 'camina', 'esta', 'existeix', 'conegut']
    accepted_output_test2 = ['va', 'camina', 'esta', 'existeix', 'conegut']
    accepted_output_test3 = ['la', 'altra', 'les', 'quina', 'nova', 'vinent']

    def test_one(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        out_one, _ = fetch(["un", "monjo", "vell", "per", "la", "muntanya"])
        assert out_one in self.accepted_output_test1

    def test_two(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        out_two, _ = fetch(["un", "monjo", "jove", "per", "la", "muntanya"])
        assert out_two in self.accepted_output_test2
    def test_three(self):
        '''The function checks whether the given input is
        contained in the corresponding accepted output'''

        out_three, _ = fetch(["el", "video", "de", "setmana", "passada", "es"])
        assert out_three in self.accepted_output_test3
