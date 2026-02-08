from .others.tbs import TBS
from .others.atbs import ATBS
from .others.tbspro import TBSPRO
from .others.tbsre import TBSRE
# from .live.pbsar import PBSARHL
from .others.ut import UT
from .others.minpmisl import MINPMISL
from .others.btst import BTSTDIR
from .others.spikesell import SPIKESELL
from .others.tbsdr import TBSDR
from .others.hedge_intraday import HEDGEINTRADAY
from .wip.putrcs import PUTRCS
from .wip.putrcscomb import PUTRCSCOMB
from .wip.pbsardelta import PBSARDELTA
from .live.pbsardelta_itmpct import PBSARITMPCT
from .wip.ptheta import PTHETA
from .live.dma import DMA
from .live.btstlive import BTSTDIRV2
from .others.pbsar_spot import PBSARSPOT
from .others.hedge_intraday_ltp import HEDGEINTRADAYLTP
from .others.hedge_overnight_ltp import HEDGEOVERNIGHTLTP
from .wip.goldcc import GOLDCC   
from .others.hedge_qaw import HEDGEQAW
from .others.pbsar_updated import PBSARHL
from .live.pbsarlegacy import PBSARLEGACY
from .wip.perpetua import PERPETUA

strats_dict = {
'atbs': ATBS,
'tbs': TBS,
'tbspro': TBSPRO,
'tbsre': TBSRE,
# 'pbsarhl': PBSARHL,
'ut': UT,
'minpmisl': MINPMISL,
'btstdir':BTSTDIR,
'spikesell': SPIKESELL,
'tbsdr': TBSDR,
'putrcs': PUTRCS,
'hedgeintraday': HEDGEINTRADAY,
'putrcscomb': PUTRCSCOMB,
'pbsardelta': PBSARDELTA,
'pbsaritmpct': PBSARITMPCT,
'ptheta': PTHETA,
'dma': DMA,
'btstdirv2': BTSTDIRV2,
'pbsarspot': PBSARSPOT,
'hedgeintradayltp':HEDGEINTRADAYLTP,
'hedgeovernightltp':HEDGEOVERNIGHTLTP,
'goldcc': GOLDCC,
'hedgeqaw': HEDGEQAW,
'pbsarhl': PBSARHL,
'pbsarlegacy': PBSARLEGACY,
'perpetua': PERPETUA
}