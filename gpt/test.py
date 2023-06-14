import os

import pandas as pd
from datasets import load_dataset

dataset = load_dataset("tiny_shakespeare")

data = dataset['train']['text']

print(os.getcwd())
df = pd.DataFrame(dataset)
print(df.info())

df.to_csv('tiny_shakespeare.csv')
print("done")

dataset = pd.read_csv('../data/tiny_shakespeare.csv')
print(dataset.info())
print(dataset['train'][0])

print("rance ta'en\nAs shall with either part's agreement stand?\n\nBAPTISTA:\nNot in my house, Lucentio; for, you know,\nPitchers have ears, and I have many servants:\nBesides, old Gremio is hearkening still;\nAnd happily we might be interrupted.\n\nTRING ha tarce: I hnatken I patow onduvestious oldleaf pare.\nFor to may,\nLes must pimion ay.\nManlotsse shad oferam faffef your shese shand\nAnk, bote'd theenl askin's but he brait:\nWhall goosn: he sacures,\nAnd mederesertinplecs' arem, \nHath not harkinise. you swo dy bloervi's so dendas forethe them:\nMeose the ke weard.\n\nRICICET:\nAdblamen.\n\nANTELBARE:\nEarst thape hevear''d herads ausef tou me end\nSorvancy upiol.\n\nIANGHANGARD:\nNt MENO:\nNard\nADd O ant, Llowtlen ath Kicke's whatce,\nWhank thenencation mo see whay.\nNend hatnat his evkellce as lids'd newen all thalt a when tirakn groecto for wiff.\n\nVINGCENTERY:\nThat: sallet o's theself shis sunte as yoth muest\nA DDWARD:\nTo filk! were all caurlesself he';\nSee latay-anser tow me, dorrcy flald bogiloss saltumeds it noon tholead leass in leakord\nInd I I hplet arow sall.-\n\nMOONIUSTINUSES:\nhesh mone: ontoo-ne ear;\nThe avenslosed.\n\nBETRU:\nse all vaseriref O sane my where ophoril.\COLORLANUSALNO:\nI laste, Is my theild rold hat so and lo, dold thou butber het may moland barartid, and vee't to at,\nI that ame\ntere sown:'\nJmle shall sto gatunte sult ou salse?\n\nDRUOEMERET:\nFARCUT:\nTo, alll asok yollow thes dastered the hisee oullford:\nI went, or genen of and foll  go\nWeravey's lasian, bray, harte nataficel and earohes woun.\n\nMORIUNELINIA:\nHateny'l the havise's broght 'eser for und, Kou mrowinds withe kes to mok\nnokn,\nHe an\nThat oute dorweldow te'd lessciught as walsio; thing susealk fader surestte coomer.\n\nKING EEDWASo:\nno won, hat you, delows epen howougherlalk I upourld my tourting boy,\nPovest knond.\nBomnand oupees ic-may may uptleeo's shosel,\nAnd feve mis, me tas yourd\n\nYocneng's? I\nSisonos?\nCast of negonde;'\nAh wilds and samsead?\n\nLAUNRC\RO:\non I is'd Ayd painy the 'n ment menm; Asally theet paitre's 'ded-havese voust life heelvot mesen's Ise wordescie:\nFow thee whithe eif as drant id aw ydost!\n\nRICIO:\'Whawer, Woullout'as fer, Oar ther makin!\nSoldy\nCarche pariedic")