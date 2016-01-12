import unittest

from trendi import paperdolls

class OutcomesTest(unittest.TestCase):
    # examples of things to return
    # def testPass(self):
    # return
    # def testFail(self):
    # self.failIf(True)
    # def testError(self):
    # raise RuntimeError('Test error!')


    def setUp(self):

        pass


    def test_blacklisted_term_in_url(self):
        goodlist=['nastygal.com',
            'cnn.com',
            'http://www.booking.com/',
            'http://ok.ru',
            'http://www.msn.com',
            'http://www.netflix.com',
            'http://vk.com',
            'http://www.dasding.de',
            'http://www.zara.com',
            'http://www.alohatube.com',
            'https://www.instagram.com',
            'http://www.jabong.com',
            'http://www.tagged.com',
            'http://www.jollychic.com',
            'http://www.aliexpress.com',
            'http://www.flipkart.com',
            'http://www.wildfashion.ro',
            'http://www.posthaus.com.br',
            'http://www.stellaswardrobe.com',
            'http://fallout4.2game.info',
            'http://www.asos.com',
            'http://www.gala.de',
            'http://shop.mango.com',
            'http://www.refinery29.com',
            'https://vk.com',
            'http://www.fashiondays.ro',
            'http://www.voonik.com',
            'http://lookbook.nu',
            'http://beeg.com',
            'http://www.freepik.com',
            'http://www.3suisses.fr',
            'http://www.clubmonaco.com',
            'http://www.amazon.com',
            'http://globoesporte.globo.com/',
            'nastygal.com']

        badlist = [
            'http://www.jizzhut.com',
            'http://www.youjizz.com',
            'http://www.sex.com',
            'http://jerkoffer.com',
            'http://xxxtubedot.com/'
            'http://www.youx.xxx/'
            'http://dirtyasiantube.com/'
            'grannyfucks.me',
            'https://www.google.co.il' ]

        for url in goodlist:
            isbad = paperdolls.blacklisted_term_in_url(url)
            print('good url {0} has badness {1}'.format(url,isbad))
            self.assertTrue(not isbad)

        for url in badlist:
            isbad = paperdolls.blacklisted_term_in_url(url)
            print('bad url {0} has badness {1}'.format(url,isbad))
            self.assertTrue(isbad)

if __name__ == "__main__":
    unittest.main()