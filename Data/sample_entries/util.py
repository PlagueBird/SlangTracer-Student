import numpy as np

class GSD_Definition:
    
    def __init__(self, def_sent):
        self.def_sent = def_sent
        
        self.stamps = []
        self.contexts = {}
        
    def add_stamp(self, time, region, quote=None):
        self.stamps.append((time, region))
        if quote is not None:
            self.contexts[(time, region)] = quote
        
    def num_stamp(self):
        return len(self.stamps)
    
    def valid(self):
        return self.num_stamp() > 0
    
    def has_context(self):
        return len(self.contexts) > 0
        
    def __str__(self):
        out_str = self.def_sent + "\n"
        for stamp in self.stamps:
            if stamp in self.contexts:
                out_str += "%s - %s - %s\n" % (str(stamp[0]), stamp[1], self.contexts[stamp])
            else:
                out_str += "%s - %s\n" % (str(stamp[0]), stamp[1])
        return out_str

class GSD_Word:
    
    def __init__(self, word, pos, homonym):
        self.word = word
        self.pos = pos
        self.homonym = homonym
        
        if homonym == 0:
            self.entry = word
        else:
            self.entry = word+'_'+str(homonym)
            
        self.definitions = []
        
        self.abbr = False
        
    def add_definition(self, definition):
        if definition.valid():
            self.definitions.append(definition)
        
    def num_def(self):
        return len(self.definitions)
    
    def set_abbr(self, flag):
        self.abbr = flag
        
    def is_abbr(self):
        return self.abbr
    
    def valid(self):
        if self.num_def() == 0:
            return False
        return np.all([d.valid() for d in self.definitions])
        
    def __str__(self):
        out_str = "[WORD]\n%s\n[POS]\n%s\n" % (self.word, self.pos)
        if self.homonym != 0:
            out_str += "[HOMONYM]\n%d\n" % self.homonym
        out_str += "[DEFINITIONS]\n"
        for d in self.definitions:
            out_str += str(d)
        return out_str
    
    def _repr_pretty_(self, p, cycle):
       p.text(str(self) if not cycle else '...')