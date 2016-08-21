import numpy as np
import re
import time
import sys
import heapq
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import threading
from io import StringIO
from multiprocessing import Pool
from multiprocessing import Lock
from multiprocessing import Manager
import queue


class ConllLine:
    def root_init(self):
        self.id = 0
        self.form = '*root*'
        self.lemma = '_'
        self.cpostag = '_'
        self.postag = '_'
        self.feats = '_'
        self.head = -1
        self.deptype = 'rroot'
        self.phead = -1
        self.pdeptype = '_'

    def __str__(self):
        return '\t'.join(
            [str(self.id), self.form, self.lemma, self.cpostag, self.postag, self.feats, str(self.head), self.deptype,
             str(self.phead), self.pdeptype])

    def __init__(self, tokens=None):
        if tokens is None:
            self.root_init()
        else:
            self.id = int(tokens[0])
            self.form = tokens[1]
            self.lemma = tokens[2]
            self.cpostag = tokens[3]
            self.postag = tokens[4]
            self.feats = tokens[5]
            self.head = int(tokens[6])
            self.deptype = tokens[7]
            if len(tokens) > 8:
                self.phead = -1 if tokens[8] == '_' else int(tokens[8])
                self.pdeptype = tokens[9]
            else:
                self.phead = -1
                self.pdeptype = '_'

    tree_line_extractor = re.compile('([a-z]+)\(.+-(\d+), (.+)-(\d+)\)')

    # stanford parser tree output:  num(Years-3, Five-1)
    def from_tree_line(self, tree_line):
        self.root_init()
        tok = self.tree_line_extractor.match(tree_line)
        self.id = int(tok.group(4))
        self.form = tok.group(3)
        self.head = int(tok.group(2))
        self.deptype = tok.group(1)


# noinspection PyUnboundLocalVariable
def read_conll(conll_file, lower):
    root = ConllLine()
    words = [root]
    for line in conll_file:
        line = line.strip()
        if len(line) > 0:
            if lower:
                line = line.lower()
            tokens = line.split('\t')
            words.append(ConllLine(tokens))
        else:
            if len(words) > 1:
                yield words
                words = [root]
    if len(tokens) > 1:
        yield tokens


def normalize(m):
    norm = np.sqrt(np.sum(m * m, axis=1))
    norm[norm == 0] = 1
    return m / norm[:, np.newaxis]


def readVocab(path):
    vocab = []
    with open(path) as f:
        for line in f:
            vocab.extend(line.strip().split())
    return dict([(w, i) for i, w in enumerate(vocab)]), vocab


# noinspection PyTypeChecker,PyTypeChecker
class Embedding:
    def __init__(self, path):
        self.m = normalize(np.load(path + '.npy'))
        self.dim = self.m.shape[1]
        self.wi, self.iw = readVocab(path + '.vocab')

    def zeros(self):
        return np.zeros(self.dim)

    def dimension(self):
        return self.dim

    def __contains__(self, w):
        return w in self.wi

    def represent(self, w):
        return self.m[self.wi[w], :]

    def scores(self, vec):
        return np.dot(self.m, vec)

    def pos_scores(self, vec):
        return (np.dot(self.m, vec) + 1) / 2

    def pos_scores2(self, vec):
        scores = np.dot(self.m, vec)
        scores[scores < 0.0] = 0.0
        return scores

    def top_scores(self, scores, n=10):
        if n <= 0:
            n = len(scores)
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])

    def closest(self, w, n=10):
        scores = np.dot(self.m, self.represent(w))
        return self.top_scores(scores, n)

    def closest_with_time(self, w, n=10):
        start = time.time()
        scores = np.dot(self.m, self.represent(w))
        end = time.time()
        #        print "\nDeltatime: %f msec\n" % ((end-start)*1000)
        return self.top_scores(scores, n), end - start

    def closest_vec(self, wordvec, n=10):
        # scores = self.m.dot(self.represent(w))
        scores = np.dot(self.m, wordvec)
        return self.top_scores(scores, n)

    #        if n <= 0:
    #            n = len(scores)
    #        return heapq.nlargest(n, zip(self.iw, scores))

    # noinspection PyTypeChecker
    def closest_vec_filtered(self, wordvec, vocab, n=10):
        scores = np.dot(self.m, wordvec)
        if n <= 0:
            n = len(scores)
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1] in vocab:
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, zip(self.iw, scores), key=lambda x: x[1])

    def closest_prefix(self, w, prefix, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1].startswith(prefix):
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])

    def closest_filtered(self, w, vocab, n=10):
        scores = np.dot(self.m, self.represent(w))
        scores_words = zip(self.iw, scores)
        for i in range(0, len(scores_words)):
            if not scores_words[i][1] in vocab:
                scores_words[i] = (-1, scores_words[i][0])
        return heapq.nlargest(n, scores_words, key=lambda x: x[1])

    def similarity(self, w1, w2):
        return self.represent(w1).dot(self.represent(w2))


# noinspection PyUnusedLocal,PyUnusedLocal
def generate_inferred(result_vec, target_word, target_lemma, pos):
    generated_word_re = re.compile('^[a-zA-Z]+$')

    generated_results = {}
    min_weight = None
    if result_vec is not None:
        for word, weight in result_vec:
            if generated_word_re.match(word) is not None:  # make sure this is not junk
                # wn_pos = to_wordnet_pos[pos]
                # lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
                if word != target_word:  # and lemma != target_lemma:
                    if word in generated_results:
                        weight = max(weight, generated_results[word])
                    generated_results[word] = weight
                    if min_weight is None:
                        min_weight = weight
                    else:
                        min_weight = min(min_weight, weight)

    if min_weight is None:
        min_weight = 0.0
    i = 0.0

    # just something to return in case not enough words were generated
    default_generated_results = ['time', 'people', 'information', 'work', 'first', 'like', 'year', 'make', 'day',
                                 'service']

    for lemma in default_generated_results:
        if len(generated_results) >= len(default_generated_results):
            break
        i -= 1.0
        generated_results[lemma] = min_weight + i

    return generated_results


def add_inference_result(token, weight, filtered_results, candidates_found):
    candidates_found.add(token)
    best_last_weight = filtered_results[token] if token in filtered_results else None
    if best_last_weight is None or weight > best_last_weight:
        filtered_results[token] = weight


class CsInferrer(object):
    """
    classdocs
    """

    def __init__(self):
        """
        Constructor
        """
        self.time = [0.0, 0]

    def inference_time(self, seconds):
        self.time[0] += seconds
        self.time[1] += 1

        # processing time in msec

    def msec_per_word(self):
        return 1000 * self.time[0] / self.time[1] if self.time[1] > 0 else 0.0


def get_deps(sent, target_ind, stopwords):
    deps = []

    for word_line in sent[1:]:
        parent_line = sent[word_line.head]
        # universal        if word_line.deptype == 'adpmod': # we are collapsing preps
        if word_line.deptype == 'prep':  # we are collapsing preps
            continue
        # universal       if word_line.deptype == 'adpobj' and parent_line.id != 0: # collapsed dependency
        if word_line.deptype == 'pobj' and parent_line.id != 0:  # collapsed dependency
            grandparent_line = sent[parent_line.head]
            if grandparent_line.id != target_ind and word_line.id != target_ind:
                continue
            relation = "%s:%s" % (parent_line.deptype, parent_line.form)
            head = grandparent_line.form
        else:  # direct dependency
            if parent_line.id != target_ind and word_line.id != target_ind:
                continue
            head = parent_line.form
            relation = word_line.deptype
        if word_line.id == target_ind:
            if head not in stopwords:
                deps.append("I_".join((relation, head)))
        else:
            if word_line.form not in stopwords:
                deps.append("_".join((relation, word_line.form)))
                #      print h,"_".join((rel,m))
                #      print m,"I_".join((rel,h))
    return deps


def vec_to_str(subvec, max_n):
    def wf2ws(weight):
        return '{0:1.5f}'.format(weight)

    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [' '.join([word, wf2ws(weight)]) for word, weight in sub_list_sorted]
    return '\t'.join(sub_strs)


class CsEmbeddingInferrer(CsInferrer):
    def __init__(self, context_math, word_path, context_path, conll_filename, top_inferences_to_analyze):

        CsInferrer.__init__(self)
        self.context_math = context_math
        self.word_vecs = Embedding(word_path)
        self.context_vecs = Embedding(context_path)
        self.use_stopwords = False
        # self.conll_file = open(conll_filename, 'r')
        # self.sents = read_conll(self.conll_file, True)
        self.top_inferences_to_analyze = top_inferences_to_analyze

        self.lemmas = {}
        for _w in self.word_vecs.iw:
            for wn_pos in [wordnet.NOUN, wordnet.ADJ, wordnet.VERB, wordnet.ADV]:
                self.lemmas['_'.join([_w, wn_pos])] = WordNetLemmatizer().lemmatize(_w, wn_pos)

    # noinspection PyShadowingNames
    def represent(self, target, deps, avg_flag, tfo):

        target_vec = None if target is None else np.copy(self.word_vecs.represent(target))
        dep_vec = None
        deps_found = 0
        for dep in deps:
            if dep in self.context_vecs:
                deps_found += 1
                if dep_vec is None:
                    dep_vec = np.copy(self.context_vecs.represent(dep))
                else:
                    dep_vec += self.context_vecs.represent(dep)
            else:
                tfo.write("NOTICE: %s not in context embeddings. Ignoring.\n" % dep)

        ret_vec = None
        if target_vec is not None:
            ret_vec = target_vec
        if dep_vec is not None:
            if avg_flag:
                dep_vec /= deps_found
            if ret_vec is None:
                ret_vec = dep_vec
            else:
                ret_vec += dep_vec

        norm = (ret_vec.dot(ret_vec.transpose())) ** 0.5
        ret_vec /= norm

        return ret_vec

    # noinspection PyShadowingNames
    def mult(self, target, deps, geo_mean_flag, tfo):

        # SUPPORT NONE TARGET

        target_vec = self.word_vecs.represent(target)
        scores = self.word_vecs.pos_scores(target_vec)
        for dep in deps:
            if dep in self.context_vecs:
                dep_vec = self.context_vecs.represent(dep)
                mult_scores = self.word_vecs.pos_scores(dep_vec)
                if geo_mean_flag:
                    mult_scores **= 1.0 / len(deps)
                scores = np.multiply(scores, mult_scores)
            else:
                tfo.write("NOTICE: %s not in context embeddings. Ignoring.\n" % dep)

        result_vec = self.word_vecs.top_scores(scores, -1)
        return result_vec

    def extract_contexts(self, lst_instance, conll):
        cur_sent = conll #next(self.sents)
        cur_sent_target_ind = lst_instance.target_ind + 1
        while cur_sent_target_ind < len(cur_sent) and cur_sent[cur_sent_target_ind].form != lst_instance.target:
            sys.stderr.write("Target word form mismatch in target id %s: %s != %s  Checking next word.\n" % (
                lst_instance.target_id, cur_sent[cur_sent_target_ind].form, lst_instance.target))
            cur_sent_target_ind += 1
        if cur_sent_target_ind == len(cur_sent):
            sys.stderr.write("Start looking backwards.\n")
            cur_sent_target_ind = lst_instance.target_ind
            while (cur_sent_target_ind > 0) and (cur_sent[cur_sent_target_ind].form != lst_instance.target):
                sys.stderr.write("Target word form mismatch in target id %s: %s != %s  Checking previous word.\n" % (
                    lst_instance.target_id, cur_sent[cur_sent_target_ind].form, lst_instance.target))
                cur_sent_target_ind -= 1
        if cur_sent_target_ind == 0:
            sys.stderr.write("ERROR: Couldn't find a match for target.")
            cur_sent_target_ind = lst_instance.target_ind + 1
        stopwords = set()
        contexts = get_deps(cur_sent, cur_sent_target_ind, stopwords)

        return contexts

    # noinspection PyShadowingNames
    def find_inferred(self, lst_instance, tfo, conll):
        lock2.acquire()
        contexts = self.extract_contexts(lst_instance, conll)
        lock2.release()

        tfo.write("Contexts for target %s are: %s\n" % (lst_instance.target, contexts))
        contexts = [c for c in contexts if c in self.context_vecs]
        tfo.write("Contexts in vocabulary for target %s are: %s\n" % (lst_instance.target, contexts))

        if lst_instance.target not in self.word_vecs:
            tfo.write("ERROR: %s not in word embeddings.Trying lemma.\n" % lst_instance.target)
            if lst_instance.target_lemma not in self.word_vecs:
                tfo.write("ERROR: lemma %s also not in word embeddings. Giving up.\n" % lst_instance.target_lemma)
                return None
            else:
                target = lst_instance.target_lemma
        else:
            target = lst_instance.target

        # 'add' and 'avg' metrics are implemented more efficiently with vector representation arithmetics
        # as shown in Omer's linguistic regularities paper,
        # this is equivalent as long as the vectors are normalized to 1
        if self.context_math == 'add':
            cs_rep = self.represent(target, contexts, False, tfo)
            if cs_rep is None:
                cs_rep = self.word_vecs.zeros()
            result_vec = self.word_vecs.closest_vec(cs_rep, -1)
        elif self.context_math == 'avg':
            cs_rep = self.represent(target, contexts, True, tfo)
            if cs_rep is None:
                cs_rep = self.word_vecs.zeros()
            result_vec = self.word_vecs.closest_vec(cs_rep, -1)
        elif self.context_math == 'mult':
            result_vec = self.mult(target, contexts, False, tfo)
        elif self.context_math == 'geomean':
            result_vec = self.mult(target, contexts, True, tfo)
        elif self.context_math == 'none':
            result_vec = self.word_vecs.closest(target, -1)
        else:
            raise Exception('Unknown context math: %s' % self.context_math)

        if result_vec is not None:
            tfo.write("Top most similar embeddings: " + vec_to_str(result_vec, self.top_inferences_to_analyze) + '\n')
        else:
            tfo.write("Top most similar embeddings: " + " contexts: None\n")

        return result_vec

    def filter_inferred(self, result_vec, candidates, pos):

        filtered_results = {}
        candidates_found = set()

        to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}

        if result_vec is not None:
            for word, weight in result_vec:
                wn_pos = to_wordnet_pos[pos]

                if '_'.join([word, wn_pos]) not in self.lemmas:
                    print('_'.join([word, wn_pos]), 'not in lemmas')

                lemma = self.lemmas['_'.join([word, wn_pos])]
                if lemma in candidates:
                    add_inference_result(lemma, weight, filtered_results, candidates_found)
                if lemma.title() in candidates:
                    add_inference_result(lemma.title(), weight, filtered_results, candidates_found)
                if word in candidates:  # there are some few cases where the candidates are not lemmatized
                    add_inference_result(word, weight, filtered_results, candidates_found)
                if word.title() in candidates:  # there are some few cases where the candidates are not lemmatized
                    add_inference_result(word.title(), weight, filtered_results, candidates_found)

                    # assign negative weights for candidates with no score
                    # they will appear last sorted according to their unigram count
                    #        candidates_left = candidates - candidates_found
                    #        for candidate in candidates_left:
                    #            count = self.w2counts[candidate] if candidate in self.w2counts else 1
                    #            score = -1 - (1.0/count) # between (-1,-2]
                    #            filtered_results[candidate] = score

        return filtered_results


# noinspection PyShadowingNames
def read_candidates(candidates_file):
    target2candidates = {}
    # finally.r::eventually;ultimately
    with open(candidates_file, 'r') as f:
        for line in f:
            segments = line.split('::')
            target = segments[0]
            candidates = set(segments[1].strip().split(';'))
            target2candidates[target] = candidates
    return target2candidates


CONTEXT_TEXT_BEGIN_INDEX = 3
TARGET_INDEX = 2
from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}


class ContextInstance(object):
    def __init__(self, line, no_pos_flag):
        """
        Constructor
        """
        self.line = line
        tokens1 = line.split("\t")
        self.target_ind = int(tokens1[TARGET_INDEX])
        self.words = tokens1[3].split()
        self.target = self.words[self.target_ind]
        self.full_target_key = tokens1[0]
        self.pos = self.full_target_key.split('.')[-1]
        self.target_key = '.'.join(self.full_target_key.split('.')[:2])  # remove suffix in cases of bar.n.v
        self.target_lemma = self.full_target_key.split('.')[0]
        self.target_id = tokens1[1]
        if self.pos in from_lst_pos:
            self.pos = from_lst_pos[self.pos]
        self.target_pos = '.'.join([self.target, '*']) if no_pos_flag is True else '.'.join([self.target, self.pos])

    def get_neighbors(self, window_size):
        tokens = self.line.split()[3:]

        if window_size > 0:
            start_pos = max(self.target_ind - window_size, 0)
            end_pos = min(self.target_ind + window_size + 1, len(tokens))
        else:
            start_pos = 0
            end_pos = len(tokens)

        neighbors = tokens[start_pos:self.target_ind] + tokens[self.target_ind + 1:end_pos]
        return neighbors

    def decorate_context(self):
        tokens = self.line.split('\t')
        words = tokens[CONTEXT_TEXT_BEGIN_INDEX].split()
        words[self.target_ind] = '__' + words[self.target_ind] + '__'
        tokens[CONTEXT_TEXT_BEGIN_INDEX] = ' '.join(words)
        return '\t'.join(tokens) + "\n"


def vec_to_str_generated(subvec, max_n):
    sub_list_sorted = heapq.nlargest(max_n, subvec, key=lambda x: x[1])
    sub_strs = [word for word, weight in sub_list_sorted]
    return ';'.join(sub_strs)



lock2 = threading.Lock()

class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

# noinspection PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames,PyShadowingNames
def thread_run(resultsfile, inferrer, target2candidates, context_lines, lock, counter):
    for context_line, conll in context_lines:
        # print(context_line)
        lst_instance = ContextInstance(context_line, False)

        tfo_s = StringIO()
        result_vec = inferrer.find_inferred(lst_instance, tfo_s, conll)

        generated_results = generate_inferred(result_vec, lst_instance.target, lst_instance.target_lemma,
                                              lst_instance.pos)

        filtered_results = inferrer.filter_inferred(result_vec, target2candidates[lst_instance.target_key],
                                                    lst_instance.pos)

        lock.acquire()
        tfo = open(resultsfile, 'a')
        tfo_ranked = open(resultsfile + '.ranked', 'a')
        tfo_generated_oot = open(resultsfile + '.generated.oot', 'a')
        tfo_generated_best = open(resultsfile + '.generated.best', 'a')

        tfo.write("\nTest context:\n")
        tfo.write("***************\n")
        tfo.write(lst_instance.decorate_context())
        tfo.write(tfo_s.getvalue())

        tfo.write("\nGenerated lemmatized results\n")
        tfo.write("***************\n")
        tfo.write("GENERATED\t" + ' '.join(
            [lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(
            generated_results.items(), 10) + "\n")
        tfo_generated_oot.write(
        ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(
            generated_results.items(), 10) + "\n")
        tfo_generated_best.write(
        ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " :: " + vec_to_str_generated(
            generated_results.items(), 1) + "\n")

        tfo.write("\nFiltered results\n")
        tfo.write("***************\n")
        tfo.write("RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(
            filtered_results.items(), len(filtered_results)) + "\n")
        tfo_ranked.write(
        "RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(
            filtered_results.items(), len(filtered_results)) + "\n")


        tfo.close()
        tfo_ranked.close()
        tfo_generated_oot.close()
        tfo_generated_best.close()
        tfo_s.close()

        counter.value += 1

        if counter.value % 10 == 0:
            print("Read %d lines" % counter.value)
        lock.release()



if __name__ == '__main__':

    oper = sys.argv[1]
    ipath = sys.argv[2]
    epath = sys.argv[3]
    opath = sys.argv[4]
    numProc = 3


    class Arg:
        pass


    args = Arg()
    args.vocabfile = None
    args.contextmath = oper

    args.embeddingpath = ipath + '/vecs'
    args.embeddingpathc = ipath + '/contexts'

    args.testfileconll = epath + '/lst_all.conll'
    args.candidatesfile = epath + '/lst.gold.candidates'
    args.testfile = epath + '/lst_all.preprocessed'

    # args.testfileconll = epath + '/coinco_all.no_problematic.sorted.conll'
    # args.candidatesfile = epath + '/coinco.no_problematic.candidates'
    # args.testfile = epath + '/coinco_all.no_problematic.preprocessed'

    args.resultsfile = opath + '/results'
    args.topgenerated = 10

    inferrer = CsEmbeddingInferrer(args.contextmath, args.embeddingpath,
                                   args.embeddingpathc, args.testfileconll, 10)

    target2candidates = read_candidates(args.candidatesfile)

    sents = read_conll(open(args.testfileconll), True)

    tfi = open(args.testfile, 'r')
    tfo = open(args.resultsfile, 'w')
    tfo_ranked = open(args.resultsfile + '.ranked', 'w')
    tfo_generated_oot = open(args.resultsfile + '.generated.oot', 'w')
    tfo_generated_best = open(args.resultsfile + '.generated.best', 'w')

    tfo.close()
    tfo_ranked.close()
    tfo_generated_oot.close()
    tfo_generated_best.close()

    pool = Pool(numProc)

    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    context_lines = ()

    lines = 0
    while True:
        context_line = tfi.readline()

        if not context_line:
            break

        lines += 1
        conll = next(sents)

        context_lines += ((context_line, conll),)

        if len(context_lines) == (2010/numProc):
            pool.apply_async(func=thread_run, args=(args.resultsfile, inferrer, target2candidates, context_lines, lock, counter))
            context_lines = ()

        # q.put(pool.apply_async(func=thread_run, args=(args.resultsfile, context_line, inferrer, conll)))


    #
    #
    #
    #
    #     tfo.write("\nTest context:\n")
    #     tfo.write("***************\n")
    #     tfo.write(lst_instance.decorate_context())
    #     tfo.write(tfo_s.getvalue())
    #
    #     #        print "end %f" % time.time()
    #     tfo_s.close()
    #
    #     if lines % 10 == 0:
    #         print("Read %d lines" % lines)
    #
    #     threading.Thread(name=lines, target=thread_run,
    #                      args=(inferrer, tfo, tfo_generated_oot, tfo_generated_best, tfo_ranked, t_completed)).start().join()
    #
    #     noinspection PyPep8
    #     while threading.active_count() > 1:
    #         time.sleep(3)
    #
    #         lst_instance = ContextInstance(context_line, False)
    #
    #         result_vec = inferrer.find_inferred(lst_instance, tfo)
    #
    #         generated_results = generate_inferred(result_vec, lst_instance.target, lst_instance.target_lemma,
    #                                                        lst_instance.pos)
    #
    #         filtered_results = inferrer.filter_inferred(result_vec, target2candidates[lst_instance.target_key],
    #                                                     lst_instance.pos)
    #
    #         tfo.write(lst_instance.decorate_context())
    #         tfo.write("\nGenerated lemmatized results\n")
    #         tfo.write("***************\n")
    #         tfo.write("GENERATED\t" + ' '.join(
    #             [lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(
    #             generated_results.items(), 10) + "\n")
    #         tfo_generated_oot.write(
    #             ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " ::: " + vec_to_str_generated(
    #                 generated_results.items(), 10) + "\n")
    #         tfo_generated_best.write(
    #             ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + " :: " + vec_to_str_generated(
    #                 generated_results.items(), 1) + "\n")
    #
    #         tfo.write("\nFiltered results\n")
    #         tfo.write("***************\n")
    #         tfo.write("RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(
    #             filtered_results.items(), len(filtered_results)) + "\n")
    #         tfo_ranked.write(
    #             "RANKED\t" + ' '.join([lst_instance.full_target_key, lst_instance.target_id]) + "\t" + vec_to_str(
    #                 filtered_results.items(), len(filtered_results)) + "\n")
    #
    #         #        print "end %f" % time.time()
    #
    #         if lines % 10 == 0:
    #             print("Read %d lines" % lines)
    #
    # while threading.active_count() > 1:
    #     time.sleep(3)

    if len(context_lines) > 0:
        pool.apply_async(func=thread_run, args=(args.resultsfile, inferrer, target2candidates, context_lines, lock, counter))

    print('all sentences queued...')

    # n = 0
    # while q.qsize() > 0:
    #     item = q.get()
    #     item.wait()
    #     n += 1
    #
    #     if n % 10 == 0:
    #         print("Read %d lines" % n)

    tfi.close()
    pool.close()
    pool.join()



