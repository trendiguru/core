__author__ = 'yonatan'
import sys

from rq import Queue

import constants
from .crawlme import scrapLinks

scrap_q = Queue('CrawlMe', connection=constants.redis_conn)
db = constants.db

fullList = {"yahoo.com", "msn.com", "yahoo.co.jp", "qq.com", "uol.com.br", "globo.com", "naver.com", "onet.pl",
             "espn.go.com", "news.yahoo.com", "163.com", "wp.pl", "sina.com.cn", "news.google.com", "bbc.co.uk",
             "cnn.com", "news.yandex.ru", "rambler.ru", "bbc.com", "cnet.com", "dailymail.co.uk", "milliyet.com.tr",
             "nytimes.com", "news.mail.ru", "zing.vn", "sports.yahoo.com", "news.yahoo.co.jp", "theguardian.com",
             "buzzfeed.com", "interia.pl", "indiatimes.com", "hurriyet.com.tr", "huffingtonpost.com", "ifeng.com",
             "t-online.de", "foxnews.com", "drudgereport.com", "sohu.com", "about.com", "weather.com", "rediff.com",
             "iqiyi.com", "bp.blogspot.com", "livedoor.jp", "repubblica.it", "wunderground.com", "vnexpress.net",
             "forbes.com", "lemonde.fr", "bloomberg.com", "mynet.com", "telegraph.co.uk", "naver.jp", "ukr.net",
             "o2.pl", "idnes.cz", "24h.com.vn", "usatoday.com", "ig.com.br", "news.163.com", "washingtonpost.com",
             "spiegel.de", "gazeta.pl", "rt.com", "gismeteo.ru", "elpais.com", "marca.com", "pixnet.net",
             "accuweather.com", "rbc.ru", "sapo.pt", "haberturk.com", "news.rambler.ru", "goo.ne.jp",
             "commentcamarche.net", "libero.it", "sozcu.com.tr", "espncricinfo.com", "clarin.com", "liveleak.com",
             "corriere.it", "news.ifeng.com", "bol.uol.com.br", "bild.de", "pantip.com", "lequipe.fr", "ynet.co.il",
             "r7.com", "lanacion.com.ar", "folha.uol.com.br", "ccm.net", "hupu.com", "elmundo.es", "rojadirecta.me",
             "bleacherreport.com", "vice.com", "news.qq.com", "sabah.com.tr", "news.sina.com.cn", "infobae.com",
             "Domain", "avon.ru", "wizaz.pl", "women.kapook.com", "minq.com", "sephora.fr", "birchbox.com",
             "totalbeauty.com", "fragrantica.com", "fragrancenet.com", "kiehls.com", "ladyfare.com", "nocibe.fr",
             "yves-rocher.fr", "lushusa.com", "beaute-test.com", "herbeauty.co", "ipsy.com", "fragrancex.com",
             "eva.ro", "fashionguide.com.tw", "byrdie.com", "jeban.com", "esteelauder.com", "allbeauty.com",
             "stitchfix.com", "shiseido.co.jp", "zdface.com", "origins.com", "lrworld.com", "skincare-univ.com",
             "primlifestyles.com", "telva.com", "nyxcosmetics.com", "thebodyshop.co.uk", "staggeringbeauty.com",
             "stylist.co.uk", "xovain.com", "purplle.com", "urcosme.com", "tartecosmetics.com", "cosme-de.com",
             "ambitiousbeauty.com", "kao.com", "prestigeladies.com", "birchbox.fr", "essence.eu", "kimiss.com",
             "vdolady.com", "coastalscents.com", "lifestylebite.com", "wahanda.com", "locatel.com.ve", "rasysa.com",
             "docebeleza.com.br", "narscosmetics.com", "hermo.my", "thebeautydepartment.com", "dailymakeover.com",
             "loreal-paris.fr", "nouveaucheap.blogspot.com", "crabtree-evelyn.com", "parfumo.de", "shiseido.com",
             "cultbeauty.co.uk", "clarinsusa.com", "youbeauty.com", "musingsofamuse.com", "laurenconrad.com",
             "shuuemura-usa.com", "escentual.com", "thebeautybox.com.br", "shopunt.com", "yslbeautyus.com",
             "elizabetharden.com", "lush.com", "blivakker.no", "thebeautyst.com", "ghdhair.com", "vichyconsult.ru",
             "urodaizdrowie.pl", "nuxe.com", "ofracosmetics.com", "rimmellondon.com", "elle.nl", "stylesatlife.com",
             "loreal-paris.es", "maccosmetics.fr", "perfume.com", "newbeauty.com", "allcosmeticswholesale.com",
             "luckyscent.com", "opi.com", "b-glowing.com", "fragrantica.es", "sundaymore.com", "jomalone.com",
             "diva.by", "ediva.gr", "bgo.com.tw", "glossybox.com", "pudelek.pl", "tmz.com", "super.cz", "ew.com",
             "entretenimento.r7.com", "hollywoodlife.com", "kapanlagi.com", "zimbio.com", "jezebel.com",
             "purepeople.com", "jeanmarcmorandini.com", "radaronline.com", "etonline.com", "voici.fr", "topito.com",
             "ciudad.com.ar", "perezhilton.com", "virgula.uol.com.br", "suggest.com", "justjared.com", "therichest.com",
             "pressroomvip.com", "dagospia.com", "closermag.fr", "pagesix.com", "digitalspy.com", "purepeople.com.br",
             "thepiratebay.uk.net", "sopitas.com", "deadline.com", "starpulse.com", "primiciasya.com", "celebuzz.com",
             "luckstars.co", "ratingcero.com", "non-stop-people.com", "tochka.net", "toofab.com", "extra.cz",
             "kozaczek.pl", "huabian.com", "bossip.com", "spletnik.ru", "wetpaint.com", "promiflash.de", "gala.fr",
             "natalie.mu", "public.fr", "digitalspy.co.uk", "bollywoodlife.com", "tiramillas.net", "storm.mg",
             "thesuperficial.com", "2sao.vn", "oprah.com", "hecklerspray.com", "gossiplankanews.com",
             "guiadasemana.com.br", "hiddenplaybook.com", "celebritynetworth.com", "vietgiaitri.com", "gossip-tv.gr",
             "okmagazine.com", "ohtuleht.ee", "alternatifim.com", "egokick.com", "relax.by", "kanyetothe.com",
             "coed.com", "tvnotas.com.mx", "lifeandstylemag.com", "hawtcelebs.com", "lazygirls.info", "sabay.com.kh",
             "hypescience.com", "wwtdd.com", "pinkvilla.com", "fusion.net", "betterfap.com", "grabo.bg",
             "madamenoire.com", "hitfix.com", "ahaonline.cz", "123telugu.com", "nofilmschool.com", "dongtw.com",
             "playgroundmag.net", "ngoisao.vn", "potins.net", "wowkeren.com", "movistar-adsl.com", "purebreak.com",
             "entertaintastic.com", "dioguinho.pt", "perfecte.ro", "napi.hu", "klubas.lt", "esmas.com", "vertele.com",
             "purebreak.com.br", "maxmodels.pl", "stylebistro.com", "fashion.ifeng.com", "tajbao.com",
             "highsnobiety.com", "osinka.ru", "trendsylvania.net", "whowhatwear.com", "fashiony.ru",
             "fashion.sina.com.cn", "lookbook.nu", "vogue.com.tw", "thefashionspot.com", "elle.com.tw",
             "vogue.com.cn", "thehunt.com", "fashionbeans.com", "models.com", "fashion.sohu.com", "purseblog.com",
             "elle.co.jp", "perfecte.md", "cosmopolitan.lt", "wwd.com", "basenotes.net", "wmagazine.com", "enrz.com",
             "moteris.lt", "styleblazer.com", "theblondesalad.com", "fashiongonerogue.com", "thesartorialist.com",
             "fashionnstyle.com", "fashion.walla.co.il", "thegloss.com", "vogue.com.au", "panele.lt",
             "collegefashion.net", "niv.ru", "desired.de", "fashionstylemag.com", "guimi.com", "fashionbank.ru",
             "vmagazine.com", "garancedore.fr", "thefashionisto.com", "fashionising.com", "modelmanagement.com",
             "mrvintage.pl", "picbazi.com", "self.com.cn", "marko.by", "lovemaegan.com", "fashiontime.ru",
             "harpersbazaar.com.hk", "stylight.com", "dietbook.biz", "harpersbazaar.co.uk", "mybodygallery.com",
             "blyzka.by", "she.com", "thecoveteur.com", "rewardstyle.com", "fustany.com", "stylishlyme.com",
             "fashion-headline.com", "9linesmag.com", "vakko.com", "fashion.telegraph.co.uk", "fashionbombdaily.com",
             "instyle.de", "stylebible.ph", "graziadaily.co.uk", "fimela.com", "brandcircus.ro", "mkstyle.net",
             "purestorm.com", "fashiontv.com", "fashionmagazine.com", "meltyfashion.fr", "avili-style.ru",
             "tech.xinmin.cn", "followme.gr", "popbee.com", "celebsvenue.com", "fashion.hola.com", "todaysfashion.com",
             "fashion.allwomenstalk.com", "hellomollyfashion.com", "magazine-data.com", "style.com",
             "wearesodroee.com", "eazyfashion.com", "candidfashionpolice.com", "malemodelscene.net", "starttoday.jp",
             "nifmagazine.com", "harpersbazaar.com.au", "levi.com.cn", "femlife.de", "tmz.com", "ew.com", "jezebel.com",
             "hollywoodlife.com", "etonline.com", "radaronline.com", "zimbio.com", "perezhilton.com", "pagesix.com",
             "suggest.com", "starpulse.com", "therichest.com", "bossip.com", "deadline.com", "toofab.com", "oprah.com",
             "hiddenplaybook.com", "pressroomvip.com", "egokick.com", "okmagazine.com", "justjared.com", "celebuzz.com",
             "wetpaint.com", "thesuperficial.com", "entertaintastic.com", "celebritynetworth.com", "madamenoire.com",
             "wwtdd.com", "buzzlamp.com", "fusion.net", "liftbump.com", "lifeandstylemag.com", "kanyetothe.com",
             "coed.com", "fishwrapper.com", "hitfix.com", "hecklerspray.com", "playbill.com", "denofgeek.us",
             "betterfap.com", "insideedition.com", "gofugyourself.com", "nofilmschool.com", "hellobeautiful.com",
             "theawl.com", "digitalspy.com", "soapcentral.com", "lazygirls.info", "wonderwall.com",
             "thesmokinggun.com", "screencrush.com", "gossipcop.com", "celebdirtylaundry.com", "the-toast.net",
             "geektyrant.com", "celebrityfanatic.com", "realitytea.com", "blindgossip.com", "celebitchy.com",
             "laineygossip.com", "maxgo.com", "pinkvilla.com", "fame10.com", "tomandlorenzo.com",
             "crazydaysandnights.net", "rumorfix.com", "justjaredjr.com", "bustedcoverage.com",
             "rumordaily.com", "comicsalliance.com", "splitsider.com", "accesshollywood.com",
             "celebritypix.com", "hollywoodtuna.com", "movieandtvcorner.com", "extratv.com",
             "facade.com", "popoholic.com", "rantchic.com", "starcasm.net", "sbux-portal.appspot.com",
             "thehollywoodgossip.com", "redlettermedia.com", "hawtcelebs.com", "closerweekly.com", "dvdtalk.com",
             "hypable.com", "soapsindepth.com", "mediamass.net", "theforce.net", "famefocus.com", "storm.mg",
             "accessthestars.com", "bollywoodlife.com", "screenjunkies.com", "starmagazine.com", "digitalspy.co.uk",
             "dayscafe.com", "barevhayer.com", "collegecandy.com", "stylebistro.com", "whowhatwear.com",
             "highsnobiety.com", "thehunt.com", "purseblog.com", "thefashionspot.com", "lookbook.nu", "wwd.com",
             "fashionbeans.com", "models.com", "wmagazine.com", "basenotes.net", "thegloss.com", "collegefashion.net",
             "fashionnstyle.com", "styleblazer.com", "mybodygallery.com", "stylight.com", "picbazi.com",
             "thesartorialist.com", "stylishlyme.com", "vogue.com.tw", "fashiongonerogue.com", "thecoveteur.com",
             "fashion.ifeng.com", "rewardstyle.com", "scoopexchange.com", "vmagazine.com", "todaysfashion.com",
             "garancedore.fr", "theblondesalad.com", "candidfashionpolice.com", "thefashionisto.com",
             "fashion.sina.com.cn", "elle.com.tw", "fashion.allwomenstalk.com", "iahfy.tumblr.com",
             "acontinuouslean.com", "fashionscope.com", "report-site.com", "modelmanagement.com",
             "fashionbombdaily.com", "stylesoftomorrow.com", "fashionising.com", "vogue.com.au",
             "fashion.telegraph.co.uk", "adriannapapell.com", "stylemined.com", "fashionstylemag.com",
             "harpersbazaar.co.uk", "lovemaegan.com", "getkempt.com", "fashion.net", "vogue.com.cn",
             "wallisfashion.com", "osinka.ru", "fashion.about.com", "wantable.com", "fimela.com", "fashion-era.com",
             "hellomollyfashion.com", "celebsvenue.com", "perfecte.md", "fashionmagazine.com", "songofstyle.com",
             "insideoutstyleblog.com", "hespokestyle.com", "styleite.com", "galadarling.com", "whatiwore.tumblr.com",
             "style.com", "kpophqpictures.co.vu", "fashionnewbie.com", "maxmodels.pl", "lcking.com",
             "lovethepinups.tumblr.com", "newsofday.com", "weddingforward.com", "thebudgetfashionista.com",
             "glamourdaze.com", "wardrobeoxygen.com", "fazmeutipo.tumblr.com", "marcel-capato.com",
             "tokyofashion.com", "fashionfixation.com", "fabulousafter40.com", "fashionisers.com",
             "frappuccini.tumblr.com", "self.com.cn", "fashionweekdaily.com", "doyoulikevintage.tumblr.com",
             "cs10.org", "nitrolicious.com", "vickumbro.tumblr.com", "malemodelscene.net", "ironandtweed.com",
             "fashiontoast.com", "flare.com", 'peonylim.com', 'vanessajackman.blogspot.co.uk', 'manrepeller.com',
             'redcarpet-fashionawards.com', 'enbrogue.com', 'wishwishwish.net', 'stellaswardrobe.com', 'bryanboy.com',
             'stylebubble.co.uk', 'pandorasykes.com', 'indtl.com', 'streetpeeper.com', '5inchandup.blogspot.co.uk',
             'bunte.de', 'nataliehartleywears.blogspot.co.uk', 'the-frugality.com', 'garypeppergirl.com',
             'weworewhat.com', 'ella-lapetiteanglaise.com', 'camilleovertherainbow.com', 'lisegrendene.com.br',
             'nadiaaboulhosn.com', 'tommyton.com', 'wearingittoday.co.uk', 'alltheprettybirds.com', 'parkandcube.com',
             'advancedstyle.blogspot.co.uk', 'disneyrollergirl.net', 'cocosteaparty.com', "tumblr.com", "index.hu",
             "as.com", 'index.hr', "akhbarak.net", "elpais.com.uy", "terra.com.br", "instagram.com", 'blick.ch',
             "filme-bune.net", 'walla.co.il', 'ynet.co.il', 'fashioncelebstyle.com', '24sata.hr', '990.ro',
             'aliexpress.com', 'ebay.com', 'elmanana.mx', 'flipkart.com', 'idens.cz', 'jutarnji.hr',
             'manoramaonline.com', 'nametests.com', 'ndtv.com', 'net.hr', 'origo.hu', 'ouedkniss.com',
             'pinoyhdreplay.org', 'pinoynetwork.se', 'postimees.ee', 'pudelek.pl', 'rakuten.co.jp', 'sapo.cv', 'tfc.tv',
             'thestar.com.my', 'trendiguru.com', 'websta.me', 'siol.net', 'tportal.hr', 'iol.pt', 'freenet.de',
             'jewelryquestion.com', 'larep.fr', 'dailymotion.com', 'mundopositivo.com.br', 'mediaset.it',
             'cifraclub.com.br'}

fashionBlogs = {"manrepeller.com", "wishwishwish.net", "parkandcube.com", "stellaswardrobe.com", "cocosteaparty.com",
               "5inchandup.blogspot.co.uk", "garypeppergirl.com", "camilleovertherainbow.com", "streetpeeper.com",
               "the-frugality.com", "disneyrollergirl.net", "weworewhat.com", "wearingittoday.co.uk",
               "ella-lapetiteanglaise.com",
               "advancedstyle.blogspot.co.uk", "indtl.com", "redcarpet-fashionawards.com", "nadiaaboulhosn.com",
               "enbrogue.com",
               "peonylim.com", "vanessajackman.blogspot.co.uk", "alltheprettybirds.com", "lisegrendene.com.br",
               "nataliehartleywears.blogspot.co.uk", "tommyton.com", "stylebubble.co.uk", "pandorasykes.com",
               "theblondesalad.com", 'notorious-mag.com',
               "thesartorialist.com", "bryanboy.com", "bunte.de", "gala.fr"}

top50Fashion = {"refinery29.com", "maxmodels.pl", "stylebistro.com", "fashion.ifeng.com", "tajbao.com",
                "highsnobiety.com", "osinka.ru", "trendsylvania.net", "whowhatwear.com", "fashiony.ru",
                "gq.com.tw", "fashion.sina.com.cn", "lookbook.nu", "vogue.com.tw", "thefashionspot.com",
                "elle.com.tw", "vogue.com.cn", "thehunt.com", "fashionbeans.com", "gqindia.com", "models.com",
                "fashion.sohu.com", "elle.co.jp", "perfecte.md", "cosmopolitan.lt", "wwd.com", "enrz.com",
                "moteris.lt", "businessoffashion.com", "styleblazer.com", "theblondesalad.com", "fashiongonerogue.com",
                "thesartorialist.com", "cupcakesandcashmere.com", "fashion.walla.co.il", "thegloss.com", "vogue.com.au",
                "panele.lt", "af-110.com", "collegefashion.net", "niv.ru", "desired.de", "fashionstylemag.com",
                "guimi.com", "fashionbank.ru", "vmagazine.com", "garancedore.fr", "thefashionisto.com",
                "fashionising.com", "modelmanagement.com"}

top50CelebSytle = {"pudelek.pl", "tmz.com", "super.cz", "ew.com", "entretenimento.r7.com", "hollywoodlife.com",
                   "kapanlagi.com", "zimbio.com", "jezebel.com", "purepeople.com", "jeanmarcmorandini.com",
                   "radaronline.com", "etonline.com", "voici.fr", "topito.com", "ciudad.com.ar", "perezhilton.com",
                   "koreaboo.com", "cztv.com", "virgula.uol.com.br", "suggest.com", "justjared.com", "therichest.com",
                   "pressroomvip.com", "dagospia.com", "closermag.fr", "kiskegyed.hu", "pagesix.com", "spynews.ro",
                   "digitalspy.com", "purepeople.com.br", "thepiratebay.uk.net", "sopitas.com", "deadline.com",
                   "starpulse.com", "multikino.pl", "zakzak.co.jp", "primiciasya.com", "celebuzz.com", "luckstars.co",
                   "ratingcero.com", "non-stop-people.com", "tochka.net", "toofab.com", "extra.cz", "kozaczek.pl",
                   "huabian.com", "bossip.com", "spletnik.ru", "wetpaint.com"}

arts_and_all_countries = {'pudelek.pl', 'tmz.com', 'super.cz', 'ew.com', 'entretenimento.r7.com', 'hollywoodlife.com',
                          'kapanlagi.com', 'zimbio.com', 'jezebel.com', 'purepeople.com', 'jeanmarcmorandini.com',
                          'radaronline.com', 'etonline.com', 'voici.fr', 'topito.com', 'ciudad.com.ar',
                          'perezhilton.com', 'virgula.uol.com.br', 'suggest.com', 'justjared.com', 'therichest.com',
                          'pressroomvip.com', 'dagospia.com', 'closermag.fr', 'pagesix.com', 'digitalspy.com',
                          'purepeople.com.br', 'thepiratebay.uk.net', 'sopitas.com', 'deadline.com', 'starpulse.com',
                          'primiciasya.com', 'celebuzz.com', 'luckstars.co', 'ratingcero.com', 'non-stop-people.com',
                          'tochka.net', 'toofab.com', 'extra.cz', 'kozaczek.pl', 'huabian.com', 'bossip.com',
                          'spletnik.ru', 'wetpaint.com', 'promiflash.de', 'gala.fr', 'natalie.mu', 'public.fr',
                          'digitalspy.co.uk', 'bollywoodlife.com', 'tiramillas.net', 'storm.mg', 'thesuperficial.com',
                          '2sao.vn', 'oprah.com', 'hecklerspray.com', 'gossiplankanews.com', 'guiadasemana.com.br',
                          'hiddenplaybook.com', 'celebritynetworth.com', 'vietgiaitri.com', 'gossip-tv.gr',
                          'okmagazine.com', 'ohtuleht.ee', 'alternatifim.com', 'egokick.com', 'relax.by',
                          'kanyetothe.com', 'coed.com', 'tvnotas.com.mx', 'lifeandstylemag.com', 'hawtcelebs.com',
                          'lazygirls.info', 'sabay.com.kh', 'hypescience.com', 'wwtdd.com', 'pinkvilla.com',
                          'fusion.net', 'betterfap.com', 'grabo.bg', 'madamenoire.com', 'hitfix.com', 'ahaonline.cz',
                          '123telugu.com', 'nofilmschool.com', 'dongtw.com', 'playgroundmag.net', 'ngoisao.vn',
                          'potins.net', 'wowkeren.com', 'movistar-adsl.com', 'purebreak.com', 'entertaintastic.com',
                          'dioguinho.pt', 'perfecte.ro', 'napi.hu', 'klubas.lt', 'esmas.com', 'vertele.com',
                          'purebreak.com.br', 'globo.com', 'telegraph.co.uk', 'filmesonlinegratis.net', 'kooora.com',
                          'yahoo-mbga.jp', 'stardoll.com', 'filmeonline2013.biz', 'filmehd.net', 'yallakora.com'}

arts_and_india = {'bollywoodlife.com', '123telugu.com', 'pinkvilla.com', 'missmalini.com', 'tollywood.net',
                  'indiancinemagallery.com', 'therichest.com', 'hollywoodlife.com', 'andhravilas.net', 'follo.in',
                  'ew.com', 'moviemint.com', 'zimbio.com', 'boxofficeindia.com', 'bollyguide.com', 'tellychowk.com',
                  'tellyduniya.com', 'bollywoodtadka.in', 'ap13.in', 'hecklerspray.com', 'starsunfolded.com',
                  'bollymeaning.com', 'radaronline.com', 'filmycity.in', 'liveinstyle.com', 'celebritytoob.com',
                  'boxofficecapsule.com', 'bollywood3.com', 'tellystars.com', 'justjared.com', 'tmz.com',
                  'thepiratebay.uk.net', 'bollybreak.in', 'foundpix.com', 'coed.com', 'screenjunkies.com',
                  'etonline.com', 'jezebel.com', 'indiglamour.com', 'nofilmschool.com', 'sara-freder.com',
                  'deadline.com', 'celebuzz.com', 'digitalspy.com', 'perezhilton.com', 'ganna.com',
                  'celebritynetworth.com', 'vietgiaitri.com', 'bollyhollynews.com', 'hitfix.com', 'sharestills.com',
                  'ndtvimg.com', 'celebritykick.com', 'hawtcelebs.com', 'yadtek.com', 'masala.com', 'denofgeek.us',
                  'djhardwell.com', 'promiflash.de', 'bharatwaves.org', 'slava.bg', 'oprah.com', 'sawfirst.com',
                  'lifeandstylemag.com', 'toofab.com', 'lazygirls.info', 'watchmojo.com', 'princemahesh.com',
                  'pressroomvip.com', 'wetpaint.com', 'screencrush.com', 'nokomis.in', 'delhievents.com',
                  'jaynestars.com', 'starpulse.com', 'celebdetail.com', 'celebritysizes.com', 'moviespicy.com',
                  'ajithfans.com', 'news.indiglamour.com', 'snehasallapam.com', 'ebharat.in', 'celebritytonic.com',
                  'okmagazine.com', 'tvactress.in', 'ratingcero.com', 'nagfans.com', 'egokick.com', 'hypable.com',
                  'boxofficeindia.co.in', 'fusion.net', 'thehollywoodgossip.com', 'aceshowbiz.com', 'nyoozflix.com',
                  'gossiplankanews.com', 'indianalerts.com', 'digitalspy.co.uk', 'indianactressphotos.in',
                  'moviepics99.com', 'stillgalaxy.com'}

arts_and_us = {'tmz.com', 'ew.com', 'jezebel.com', 'hollywoodlife.com', 'etonline.com', 'radaronline.com', 'zimbio.com',
               'perezhilton.com', 'pagesix.com', 'suggest.com', 'starpulse.com', 'therichest.com', 'bossip.com',
               'deadline.com', 'toofab.com', 'oprah.com', 'hiddenplaybook.com', 'pressroomvip.com', 'egokick.com',
               'okmagazine.com', 'justjared.com', 'celebuzz.com', 'wetpaint.com', 'thesuperficial.com',
               'entertaintastic.com', 'celebritynetworth.com', 'madamenoire.com', 'wwtdd.com', 'buzzlamp.com',
               'fusion.net', 'liftbump.com', 'lifeandstylemag.com', 'kanyetothe.com', 'coed.com', 'fishwrapper.com',
               'hitfix.com', 'hecklerspray.com', 'playbill.com', 'denofgeek.us', 'betterfap.com', 'insideedition.com',
               'gofugyourself.com', 'nofilmschool.com', 'hellobeautiful.com', 'theawl.com', 'digitalspy.com',
               'soapcentral.com', 'lazygirls.info', 'wonderwall.com', 'thesmokinggun.com', 'screencrush.com',
               'gossipcop.com', 'celebdirtylaundry.com', 'the-toast.net', 'geektyrant.com', 'celebrityfanatic.com',
               'realitytea.com', 'blindgossip.com', 'celebitchy.com', 'laineygossip.com', 'maxgo.com', 'pinkvilla.com',
               'fame10.com', 'tomandlorenzo.com', 'crazydaysandnights.net', 'rumorfix.com', 'justjaredjr.com',
               'bustedcoverage.com', 'rumordaily.com', 'comicsalliance.com', 'splitsider.com', 'accesshollywood.com',
               'celebritypix.com', 'hollywoodtuna.com', 'movieandtvcorner.com', 'extratv.com', 'facade.com',
               'popoholic.com', 'rantchic.com', 'starcasm.net', 'sbux-portal.appspot.com', 'thehollywoodgossip.com',
               'redlettermedia.com', 'hawtcelebs.com', 'closerweekly.com', 'dvdtalk.com', 'hypable.com',
               'soapsindepth.com', 'mediamass.net', 'theforce.net', 'famefocus.com', 'storm.mg', 'accessthestars.com',
               'bollywoodlife.com', 'screenjunkies.com', 'starmagazine.com', 'digitalspy.co.uk', 'dayscafe.com',
               'barevhayer.com', 'collegecandy.com'}

fash_and_mod_all = {'maxmodels.pl', 'stylebistro.com', 'fashion.ifeng.com', 'tajbao.com', 'highsnobiety.com',
                    'osinka.ru', 'trendsylvania.net', 'whowhatwear.com', 'fashiony.ru', 'fashion.sina.com.cn',
                    'lookbook.nu', 'vogue.com.tw', 'thefashionspot.com', 'elle.com.tw', 'vogue.com.cn', 'thehunt.com',
                    'fashionbeans.com', 'models.com', 'fashion.sohu.com', 'purseblog.com', 'elle.co.jp', 'perfecte.md',
                    'cosmopolitan.lt', 'wwd.com', 'basenotes.net', 'wmagazine.com', 'enrz.com', 'moteris.lt',
                    'styleblazer.com', 'theblondesalad.com', 'fashiongonerogue.com', 'thesartorialist.com',
                    'fashionnstyle.com', 'fashion.walla.co.il', 'thegloss.com', 'vogue.com.au', 'panele.lt',
                    'collegefashion.net', 'niv.ru', 'desired.de', 'fashionstylemag.com', 'guimi.com', 'fashionbank.ru',
                    'vmagazine.com', 'garancedore.fr', 'thefashionisto.com', 'fashionising.com', 'modelmanagement.com',
                    'mrvintage.pl', 'picbazi.com', 'self.com.cn', 'marko.by', 'lovemaegan.com', 'fashiontime.ru',
                    'harpersbazaar.com.hk', 'stylight.com', 'dietbook.biz', 'harpersbazaar.co.uk', 'mybodygallery.com',
                    'blyzka.by', 'she.com', 'thecoveteur.com', 'rewardstyle.com', 'fustany.com', 'stylishlyme.com',
                    'fashion-headline.com', '9linesmag.com', 'vakko.com', 'fashion.telegraph.co.uk',
                    'fashionbombdaily.com', 'instyle.de', 'stylebible.ph', 'graziadaily.co.uk', 'fimela.com',
                    'brandcircus.ro', 'mkstyle.net', 'purestorm.com', 'fashiontv.com', 'fashionmagazine.com',
                    'meltyfashion.fr', 'avili-style.ru', 'tech.xinmin.cn', 'followme.gr', 'popbee.com',
                    'celebsvenue.com', 'fashion.hola.com', 'todaysfashion.com', 'fashion.allwomenstalk.com',
                    'hellomollyfashion.com', 'magazine-data.com', 'style.com', 'wearesodroee.com', 'eazyfashion.com',
                    'candidfashionpolice.com', 'malemodelscene.net', 'starttoday.jp', 'nifmagazine.com',
                    'harpersbazaar.com.au', 'levi.com.cn', 'femlife.de'}

fash_and_mod_india = {'tajbao.com', 'roposo.com', 'styleblazer.com', 'whowhatwear.com', 'stylebistro.com',
                      'fashionbeans.com', 'highsnobiety.com', 'fashiontv.com', 'lovemaegan.com', 'thefashionspot.com',
                      'fashionnstyle.com', 'basenotes.net', 'lookbook.nu', 'thefashionisto.com', 'models.com',
                      'collegefashion.net', 'thegloss.com', 'fashiongonerogue.com', 'thehunt.com',
                      'theamazingmodels.com', 'fashionstylemag.com', 'wwd.com', 'modelmanagement.com',
                      'fashionbombdaily.com', 'purseblog.com', 'fashionwtf.com', 'fashion.allwomenstalk.com',
                      'fashionscandal.com', 'lakmefashionweek.co.in', 'fashionvalley.in', 'dressanarkali.com',
                      'styleinked.com', 'vogue.com.au', 'wmagazine.com', 'fashion.mithilaconnect.com',
                      'indianmalemodels.me', 'globalbeauties.com', 'tophighfashion.org', 'stylecracker.com',
                      'chaahatfashionjewellery.com', 'heartifb.com', 'fashionwal.com', 'theblondesalad.com',
                      'style.pk', 'fashionunited.com', 'stylehuntworld.blogspot.in', 'thesartorialist.com',
                      'fashioncentral.pk', 'tellypedia.com', 'stylishbynature.com', 'sabyasachi.com',
                      'handfulofshadows.wordpress.com', 'rusolclothing.com', 'fashionmagazine.com', 'thecoveteur.com',
                      'myfashionvilla.com', 'ritikagulati.com', 'fabfashionfix.com',
                      'ksheerabdidwadasivrata.blogspot.in', 'fashionisers.com', 'fashionising.com', 'celebsvenue.com',
                      'stylishlyme.com', 'fashionfad.in', 'vmagazine.com', 'tokyofashion.com', 'glamourdaze.com',
                      'fashion-era.com', 'mtifashion.com', 'fashion.telegraph.co.uk', 'letsexpresso.com',
                      'ebuzzfashion.com', 'gngmodels.com', 'fashionworld-hamood.blogspot.in',
                      'thebudgetfashionista.com', 'riteeriwajethnic.com', 'fashionmaxi.com', 'fashiontrendspk.com',
                      'fashion.about.com', 'harpersbazaar.com.au', 'fashionstown.com', 'galadarling.com',
                      'hollybollyscoop.com', 'thefashionflite.com', 'flare.com', 'purplemodelmanagement.com',
                      'fashiontrendsetter.com', 'dreeamcast.com', 'golpobd24.blogspot.in', 'styldrv.com',
                      'fashionpr.com', 'shefashiontrend.com', 'blender-models.com', 'vintage-obsession.com',
                      'fashion.com', 'd2xzbo2ehns0vv.cloudfront.net', 'malemodelscene.net', 'fashionindie.com',
                      'fustany.com', 'minhaz.me'}

fash_and_mod_us = {'stylebistro.com', 'whowhatwear.com', 'highsnobiety.com', 'thehunt.com', 'purseblog.com',
                   'thefashionspot.com', 'lookbook.nu', 'wwd.com', 'fashionbeans.com', 'models.com', 'wmagazine.com',
                   'basenotes.net', 'thegloss.com', 'collegefashion.net', 'fashionnstyle.com', 'styleblazer.com',
                   'mybodygallery.com', 'stylight.com', 'picbazi.com', 'thesartorialist.com', 'stylishlyme.com',
                   'vogue.com.tw', 'fashiongonerogue.com', 'thecoveteur.com', 'fashion.ifeng.com', 'rewardstyle.com',
                   'scoopexchange.com', 'vmagazine.com', 'todaysfashion.com', 'garancedore.fr', 'theblondesalad.com',
                   'candidfashionpolice.com', 'thefashionisto.com', 'fashion.sina.com.cn', 'elle.com.tw',
                   'fashion.allwomenstalk.com', 'iahfy.tumblr.com', 'acontinuouslean.com', 'fashionscope.com',
                   'report-site.com', 'modelmanagement.com', 'fashionbombdaily.com', 'stylesoftomorrow.com',
                   'fashionising.com', 'vogue.com.au', 'fashion.telegraph.co.uk', 'adriannapapell.com',
                   'stylemined.com', 'fashionstylemag.com', 'harpersbazaar.co.uk', 'lovemaegan.com', 'getkempt.com',
                   'fashion.net', 'vogue.com.cn', 'wallisfashion.com', 'osinka.ru', 'fashion.about.com', 'wantable.com',
                   'fimela.com', 'fashion-era.com', 'hellomollyfashion.com', 'celebsvenue.com', 'perfecte.md',
                   'fashionmagazine.com', 'songofstyle.com', 'insideoutstyleblog.com', 'hespokestyle.com',
                   'styleite.com', 'galadarling.com', 'whatiwore.tumblr.com', 'style.com', 'kpophqpictures.co.vu',
                   'fashionnewbie.com', 'maxmodels.pl', 'lcking.com', 'lovethepinups.tumblr.com', 'newsofday.com',
                   'weddingforward.com', 'thebudgetfashionista.com', 'glamourdaze.com', 'wardrobeoxygen.com',
                   'fazmeutipo.tumblr.com', 'marcel-capato.com', 'tokyofashion.com', 'fashionfixation.com',
                   'fabulousafter40.com', 'fashionisers.com', 'frappuccini.tumblr.com', 'self.com.cn',
                   'fashionweekdaily.com', 'doyoulikevintage.tumblr.com', 'cs10.org', 'nitrolicious.com',
                   'vickumbro.tumblr.com', 'malemodelscene.net', 'ironandtweed.com', 'fashiontoast.com', 'flare.com',
                   'heartifb.com', 'mylifeaseva.com'}

e_zines = {'buzzfeed.com', 'forbes.com', 'mynet.com', 'pixnet.net', 'vice.com', 'novinky.cz', 'rollingstone.com',
           'refinery29.com', 'people.com', 'focus.de', 'complex.com', 'time.com', 'theatlantic.com', 'obozrevatel.com',
           'xl.pt', 'littlethings.com', 'slate.com', 'si.com', 'lepoint.fr', 'blic.rs', 'korrespondent.net',
           'upsocl.com', 'vogue.com', 'esquire.com', 'salon.com', 'askmen.com', 'censor.net.ua', 'intoday.in',
           'nature.com', 'stern.de', 'blogtamsu.vn', 'vz.ru', 'sinaimg.cn', 'realsimple.com', 'cosmopolitan.com',
           'usnews.com', 'vox.com', 'timeout.com', 'nymag.com', 'distractify.com', 'newyorker.com', 'variety.com',
           'harpersbazaar.com', 'elle.com', 'gq.com', 'aplus.com', 'inc.com', 'bizjournals.com', 'dose.com',
           'gentside.com', 'libertatea.ro', 'vanityfair.com', 'telegraf.com.ua', 'elle.fr', 'imujer.com', 'dir.bg',
           'caras.uol.com.br', 'somethingawful.com', 'batanga.com', 'thesun.co.uk', 'thrillist.com',
           'opposingviews.com',
           'seventeen.com', 'motherjones.com', 'pjmedia.com', 'meneame.net', 'hypebeast.com', 'tomshw.it',
           'fastcompany.com', 'fanpage.gr', 'svpressa.ru', 'parismatch.com', 'marieclaire.com', 'theweek.com',
           'tiphero.com', 'prozeny.cz', 'dnaindia.com', 'srbijadanas.com', 'sopitas.com', 'fontanka.ru', 'kiskegyed.hu',
           'adweek.com', 'redbookmag.com', 'nerdist.com', 'test.de', 'allure.com', 'cosmo.ru', 'newsweek.com',
           'bunte.de', 'boldsky.com', 'gunosy.com', 'instyle.com', 'naturalnews.com', 'pronto.com.ar', 'barrons.com',
           'madmoizelle.com', 'glamour.com', 'walkerplus.com', 'maxim.com', 'runnersworld.com'}

all_white_lists = frozenset().union(fullList, fashionBlogs, top50CelebSytle, top50Fashion, arts_and_all_countries,
                                    arts_and_india, arts_and_us, fash_and_mod_all, fash_and_mod_india, fash_and_mod_us,
                                    e_zines)


def masterCrawler(floor=2, whiteList=top50Fashion):
    db.crawler_processed.drop()
    db.crawler_processed.create_index("url")
    for site in whiteList:
        url = "http://www." + site
        scrap_q.enqueue(scrapLinks, url, floor)
    return "finished"


if __name__ == "__main__":
    print ("Scraping the white list - Started...)")
    levels = 2
    whiteLi = top50Fashion
    if len(sys.argv) > 1:
        levels = int(sys.argv[1])
    if len(sys.argv) > 2:
        if sys.argv[2] == "top50CelebSytle":
            whiteLi = top50CelebSytle
        elif sys.argv[2] == "fashionBlogs":
            whiteLi = fashionBlogs
        elif sys.argv[2] == "fullList":
            whiteLi = fullList
        else:
            whiteLi = [sys.argv[2]]
    res = masterCrawler(levels, whiteLi)
    print (res)
