import random
import re
import enum
from types import ModuleType
from dataclasses import dataclass
from typing import (
    Union,
    Optional,
    List,
    Dict,
    Any,
    Type,
    TextIO,
    Tuple,
    IO,
    Iterable,
    Set,
    Iterator,
)
import os
from elevenlabs.client import ElevenLabs
from elevenlabs.play import save
import librosa
import soundfile as sf
from typing import List

from soundsright.base.templates import (
    VERBS_LIST,
    NOUNS_LIST,
    ADJECTIVES_LIST,
)
from soundsright.base.utils import subnet_logger 

from dotenv import load_dotenv 
load_dotenv() 

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

### The following is a modified form of the RandomWord and RandomSentence classes from the wonderwords module: https://github.com/mrmaxguns/wonderwordsmodule
### Please reference the LICENSE and THIRD_PARTY_LICENSES files for more information.
class NoWordsToChooseFrom(Exception):
    """NoWordsToChooseFrom is raised when there is an attempt to access more
    words than exist. This exception may be raised if the amount of random
    words to generate is larger than the amount of words that exist.
    """

    pass


class Defaults(enum.Enum):
    """This enum represents the default word lists. For example, if you want a
    random word generator with one category labeled 'adj' for adjectives, but
    want to use the default word list, you can do the following::

        >>> from wonderwords import RandomWord, Defaults
        >>> w = RandomWord(adj=Defaults.ADJECTIVES)
        >>> w.word()
        'red'

    Options available:

    * ``Defaults.NOUNS``: Represents a list of nouns
    * ``Defaults.VERBS``: Represents a list of verbs
    * ``Defaults.ADJECTIVES``: Represents a list of adjectives
    * ``Defaults.PROFANITIES``: Represents a list of curse words

    """

    NOUNS = NOUNS_LIST
    VERBS = VERBS_LIST
    ADJECTIVES = ADJECTIVES_LIST
    PROFANITIES = []

VOWELS = ["a", "e", "i", "o", "u"]
WordList = List[str]

# A dictionary where each key representing a category like 'nouns' corresponds to a list of words.
_DEFAULT_CATEGORIES: Dict[Defaults, WordList] = {
    Defaults.NOUNS: NOUNS_LIST,
    Defaults.VERBS: VERBS_LIST,
    Defaults.ADJECTIVES: ADJECTIVES_LIST,
    Defaults.PROFANITIES: [],
}

def is_profanity(word: str) -> bool:
    """See if a word matches one of the words in the profanity list.

    :param word: The word to check
    :return: Whether the word was found in the profanity list
    """
    return False


def filter_profanity(words: Iterable[str]) -> Iterator[str]:
    """Attempt to filter out profane words from a list. This should be done in all user-facing applications if random
    words are generated to avoid anything that could possibly be offensive. Curse word filtering is currently not done
    by default on the :py:class:`RandomWord` class.

    Example::

        >>> from wonderwords import filter_profanity
        >>> list(filter_profanity(["hello", "aSS", "world"]))
        ["hello", "world"]

    :param words: Iterable of words to filter
    :return: An iterable of the filtered result
    """
    return words

def _present_tense(verb: str) -> str:
    """Convert a verb from the infinitive to the present tense 3rd person
    form"""
    verb = verb.strip().lower()
    if verb.endswith(("ss", "ch", "x", "tch", "sh", "zz")):
        verb = verb + "es"
    elif verb.endswith("y") and not verb.endswith(
        tuple([vowel + "y" for vowel in VOWELS])
    ):
        verb = verb[:-1] + "ies"
    else:
        verb = verb + "s"
    return verb


def _with_article(word: str, rng: random.Random) -> str:
    (article,) = rng.choices(["the", "a", ""], weights=[5, 3, 2])
    if article == "a" and word[0] in VOWELS:
        article = "an"
    if article:
        article += " "
    return f"{article}{word}"

### RandomWord adaptation

class RandomWord:
    """The RandomWord class encapsulates multiple methods dealing with the
    generation of random words and lists of random words.

    Example::

        >>> from wonderwords import RandomWord, Defaults
        >>>
        >>> r = RandomWord(noun=["apple", "orange"]) # Category 'noun' with
        ...     # the words 'apple' and 'orange'
        >>> r2 = RandomWord() # Use the default word lists
        >>> r3 = RandomWord(noun=Defaults.NOUNS) # Set the category 'noun' to
        ...     # the default list of nouns

    .. important::

       Wonderwords version ``2.0`` does not have custom
       categories. In fact there are only three categories: nouns, verbs, and
       adjectives. However, wonderwords will remain backwards compatible until
       version ``3``. Please note that the ``parts_of_speech`` attribute will
       soon be deprecated, along with other method-specific features.

    :param enhanced_prefixes: whether to internally use a trie data
        structure to speed up ``starts_with`` and ``ends_with``. If enabled,
        the class takes longer to instantiate, but calls to the generation
        functions will be significantly (up to 4x) faster when using the
        ``starts_with`` and ``ends_with`` arguments. Defaults to True.
    :type enhanced_prefixes: bool, optional
    :param rng: an instance of a ``random.Random`` used for randomization
    :type rng: random.Random, optional
    :param kwargs: keyword arguments where each key is a category of words
        and value is a list of words in that category. You can also use a
        default list of words by using a value from the `Default` enum instead.
    :type kwargs: list, optional

    """

    def __init__(
        self, rng=None, **kwargs: Union[WordList, Defaults]
    ):
        # A dictionary where lists of words organized into named categories
        self._categories: Dict[str, WordList]
        # If enhanced prefixes are enabled, these tries represent all the words known to the generator in forward and
        # reverse. If disabled, this is just None.
        # Random number generator.
        self._generator: random.Random = rng or random.Random()
        # Kept for backwards compatibility. Same as self._categories
        self.parts_of_speech: Dict[str, WordList]

        if kwargs:
            self._categories = self._get_word_lists_by_category(kwargs)
        else:
            self._categories = self._get_word_lists_by_category(
                {
                    "noun": Defaults.NOUNS,
                    "verb": Defaults.VERBS,
                    "adjective": Defaults.ADJECTIVES,
                    # The following was added for backwards compatibility
                    # reasons. The plural categories will be deleted in
                    # wonderwords version 3. See issue #9.
                    "nouns": Defaults.NOUNS,
                    "verbs": Defaults.VERBS,
                    "adjectives": Defaults.ADJECTIVES,
                }
            )

        self._tries = None

        self.parts_of_speech = self._categories

    def filter(  # noqa: C901
        self,
        starts_with: str = "",
        ends_with: str = "",
        include_categories: Optional[Iterable[str]] = None,
        include_parts_of_speech: Optional[Iterable[str]] = None,
        word_min_length: Optional[int] = None,
        word_max_length: Optional[int] = None,
        regex: Optional[str] = None,
        exclude_with_spaces: bool = False,
    ) -> WordList:
        """Return a sorted list of all existing words that match the criteria
        specified by the arguments.

        Example::

            >>> # Filter all nouns that start with a:
            >>> r.filter(starts_with="a", include_categories=["noun"])

        .. important:: The ``include_parts_of_speech`` argument will soon be
            deprecated. Use ``include_categories`` which performs the exact
            same role.

        :param starts_with: the string each word should start with. Defaults to
            "".
        :type starts_with: str, optional
        :param ends_with: the string each word should end with. Defaults to "".
        :type ends_with: str, optional
        :param include_categories: a list of strings denoting a part of
            speech. Each word returned will be in the category of at least one
            part of speech. By default, all parts of speech are enabled.
            Defaults to None.
        :type include_categories: list of strings, optional
        :param include_parts_of_speech: Same as include_categories, but will
            soon be deprecated.
        :type include_parts_of_speech: list of strings, optional
        :param word_min_length: the minimum length of each word. Defaults to
            None.
        :type word_min_length: int, optional
        :param word_max_length: the maximum length of each word. Defaults to
            None.
        :type word_max_length: int, optional
        :param regex: a custom regular expression which each word must fully
            match (re.fullmatch). Defaults to None.
        :type regex: str, optional
        :param exclude_with_spaces: exclude words that may have spaces in them
        :type exclude_with_spaces: bool, optional

        :return: a list of unique words that match each of the criteria
            specified
        :rtype: list of strings
        """
        word_min_length, word_max_length = self._validate_lengths(
            word_min_length, word_max_length
        )

        # include_parts_of_speech will be deprecated in a future release
        if not include_categories:
            if include_parts_of_speech:
                include_categories = include_parts_of_speech
            else:
                include_categories = list(self._categories.keys())

        # Filter by part of speech and length. Both of these things
        # are done at once since categories are specifically ordered
        # in order to make filtering by length an efficient process.
        # See issue #14 for details.
        words = set()

        for category in include_categories:
            try:
                words_in_category = self._categories[category]
            except KeyError:
                raise ValueError(f"'{category}' is an invalid category") from None

            words_to_add = self._get_words_of_length(
                words_in_category, word_min_length, word_max_length
            )
            words.update(words_to_add)

        if self._tries is not None:
            if starts_with:
                words = words & self._tries[0].get_words_that_start_with(starts_with)
            if ends_with:
                # Since the ends_with trie is in reverse, the
                # ends_with variable must also be reversed.
                # Example (apple):
                # - Backwards: elppa
                # - ends_with: el
                # Currently this is very clunky, since all words
                # that match then need to be reversed to their
                # original forms. Currently, I have no idea how
                # to improve this. But, even with the extra overhead
                # of iteration, this system still significantly
                # shortens the amount of time to filter the words.
                ends_with = ends_with[::-1]
                words = words & set(
                    [
                        i[::-1]
                        for i in self._tries[1].get_words_that_start_with(ends_with)
                    ]
                )

        # Long operations that require looping over every word
        # (O(n)). Since they are so time-consuming, the arguments
        # passed to the function are first checked if the user
        # actually specified any time-consuming arguments. If they
        # are, all long filters are checked at once for every word.
        long_operations: Dict[str, Any] = {}

        if regex is not None:
            long_operations["regex"] = re.compile(regex)
        if exclude_with_spaces:
            long_operations["exclude_with_spaces"] = None
        if self._tries is None:
            if starts_with:
                long_operations["starts_with"] = starts_with
            if ends_with:
                long_operations["ends_with"] = ends_with

        if long_operations:
            words -= self._perform_long_operations(words, long_operations)

        return sorted(list(words))

    def random_words(
        self,
        amount: int = 1,
        starts_with: str = "",
        ends_with: str = "",
        include_categories: Optional[Iterable[str]] = None,
        include_parts_of_speech: Optional[Iterable[str]] = None,
        word_min_length: Optional[int] = None,
        word_max_length: Optional[int] = None,
        regex: Optional[str] = None,
        return_less_if_necessary: bool = False,
        exclude_with_spaces: bool = False,
    ) -> WordList:
        """Generate a list of n random words specified by the ``amount``
        parameter and fit the criteria specified.

        Example::

            >>> # Generate a list of 3 adjectives or nouns which start with
            ...     # "at"
            >>> # and are at least 2 letters long
            >>> r.random_words(
            ...     3,
            ...     starts_with="at",
            ...     include_parts_of_speech=["adjectives", "nouns"],
            ...     word_min_length=2
            ... )

        :param amount: the amount of words to generate. Defaults to 1.
        :type amount: int, optional
        :param starts_with: the string each word should start with. Defaults to
            "".
        :type starts_with: str, optional
        :param ends_with: the string each word should end with. Defaults to "".
        :type ends_with: str, optional
        :param include_categories: a list of strings denoting a part of
            speech. Each word returned will be in the category of at least one
            part of speech. By default, all parts of speech are enabled.
            Defaults to None.
        :type include_categories: list of strings, optional
        :param include_parts_of_speech: Same as include_categories, but will
            soon be deprecated.
        :type include_parts_of_speech: list of strings, optional
        :param word_min_length: the minimum length of each word. Defaults to
            None.
        :type word_min_length: int, optional
        :param word_max_length: the maximum length of each word. Defaults to
            None.
        :type word_max_length: int, optional
        :param regex: a custom regular expression which each word must fully
            match (re.fullmatch). Defaults to None.
        :type regex: str, optional
        :param return_less_if_necessary: if set to True, if there aren't enough
            words to statisfy the amount, instead of raising a
            NoWordsToChoseFrom exception, return all words that did statisfy
            the original query.
        :type return_less_if_necessary: bool, optional
        :param exclude_with_spaces: exclude words that may have spaces in them
        :type exclude_with_spaces: bool, optional

        :raises NoWordsToChoseFrom: if there are fewer words to choose from than
            the amount that was requested, a NoWordsToChoseFrom exception is
            raised, **unless** return_less_if_necessary is set to True.

        :return: a list of the words
        :rtype: list of strings
        """
        choose_from = self.filter(
            starts_with=starts_with,
            ends_with=ends_with,
            include_categories=include_categories,
            include_parts_of_speech=include_parts_of_speech,
            word_min_length=word_min_length,
            word_max_length=word_max_length,
            regex=regex,
            exclude_with_spaces=exclude_with_spaces,
        )

        if len(choose_from) < amount:
            if return_less_if_necessary:
                self._generator.shuffle(choose_from)
                return choose_from
            else:
                raise NoWordsToChooseFrom(
                    "There aren't enough words to choose from. Cannot generate "
                    f"{str(amount)} word(s)"
                )

        words = []
        for _ in range(amount):
            new_word = self._generator.choice(choose_from)
            choose_from.remove(new_word)
            words.append(new_word)

        return words

    def word(
        self,
        starts_with: str = "",
        ends_with: str = "",
        include_categories: Optional[Iterable[str]] = None,
        include_parts_of_speech: Optional[Iterable[str]] = None,
        word_min_length: Optional[int] = None,
        word_max_length: Optional[int] = None,
        regex: Optional[str] = None,
        exclude_with_spaces: bool = False,
    ) -> str:
        """Returns a random word that fits the criteria specified by the
        arguments.

        Example::

            >>> # Select a random noun that starts with y
            >>> r.word(ends_with="y", include_parts_of_speech=["nouns"])

        :param starts_with: the string each word should start with. Defaults to
            "".
        :type starts_with: str, optional
        :param ends_with: the string the word should end with. Defaults to "".
        :type ends_with: str, optional
        :param include_categories: a list of strings denoting a part of
            speech. Each word returned will be in the category of at least one
            part of speech. By default, all parts of speech are enabled.
            Defaults to None.
        :type include_categories: list of strings, optional
        :param include_parts_of_speech: Same as include_categories, but will
            soon be deprecated.
        :type include_parts_of_speech: list of strings, optional
        :param word_min_length: the minimum length of the word. Defaults to
            None.
        :type word_min_length: int, optional
        :param word_max_length: the maximum length of the word. Defaults to
            None.
        :type word_max_length: int, optional
        :param regex: a custom regular expression which each word must fully
            match (re.fullmatch). Defaults to None.
        :type regex: str, optional
        :param exclude_with_spaces: exclude words that may have spaces in them
        :type exclude_with_spaces: bool, optional

        :raises NoWordsToChoseFrom: if a word fitting the criteria doesn't
            exist

        :return: a word
        :rtype: str
        """
        return self.random_words(
            amount=1,
            starts_with=starts_with,
            ends_with=ends_with,
            include_categories=include_categories,
            include_parts_of_speech=include_parts_of_speech,
            word_min_length=word_min_length,
            word_max_length=word_max_length,
            regex=regex,
            exclude_with_spaces=exclude_with_spaces,
        )[0]

    def _validate_lengths(
        self, word_min_length: Any, word_max_length: Any
    ) -> Tuple[Union[int, None], Union[int, None]]:
        """Validate the values and types of word_min_length and word_max_length"""
        if not isinstance(word_min_length, (int, type(None))):
            raise TypeError("word_min_length must be type int or None")

        if not isinstance(word_max_length, (int, type(None))):
            raise TypeError("word_max_length must be type int or None")

        if word_min_length is not None and word_max_length is not None:
            if word_min_length > word_max_length != 0:
                raise ValueError(
                    "word_min_length cannot be greater than word_max_length"
                )

        if word_min_length is not None and word_min_length < 0:
            word_min_length = None

        if word_max_length is not None and word_max_length < 0:
            word_max_length = None

        return word_min_length, word_max_length

    def _get_word_lists_by_category(
        self, custom_categories: Dict[str, Any]
    ) -> Dict[str, WordList]:
        """Add custom categories of words"""
        out = {}
        for name, words in custom_categories.items():
            if isinstance(words, Defaults):
                word_list = _DEFAULT_CATEGORIES[words]
            else:
                word_list = words

            # All the words in each category are sorted. This is so
            # that they can be bisected by length later on for more
            # efficient word length retrieval. See issue #14 for
            # details.
            word_list.sort(key=len)
            out[name] = word_list

        return out

    def _get_words_of_length(
        self,
        word_list: WordList,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> WordList:
        """Given ``word_list``, get all words that are at least
        ``min_length`` long and at most ``max_length`` long.
        """

        if min_length is None:
            left_index = 0
        else:
            left_index = self._bisect_by_length(word_list, min_length)

        if max_length is None:
            right_index = None
        else:
            right_index = self._bisect_by_length(word_list, max_length + 1)

        return word_list[left_index:right_index]

    def _bisect_by_length(self, words: WordList, target_length: int) -> int:
        """Given a list of sorted words by length, get the index of the
        first word that's of the ``target_length``.
        """

        left = 0
        right = len(words) - 1

        while left <= right:
            middle = left + (right - left) // 2
            if len(words[middle]) < target_length:
                left = middle + 1
            else:
                right = middle - 1

        return left

    def _perform_long_operations(
        self, words: Set[str], long_operations: Dict[str, Any]
    ) -> Set[str]:
        """Return a set of words that do not meet the criteria specified by the long operations."""
        remove_words = set()
        for word in words:
            if "regex" in long_operations:
                if not long_operations["regex"].fullmatch(word):
                    remove_words.add(word)
            if "exclude_with_spaces" in long_operations:
                if " " in word:
                    remove_words.add(word)
            if "starts_with" in long_operations:
                if not word.startswith(long_operations["starts_with"]):
                    remove_words.add(word)
            if "ends_with" in long_operations:
                if not word.endswith(long_operations["ends_with"]):
                    remove_words.add(word)
        return remove_words

### RandomSentence adaptation

class RandomSentence:
    """The RandomSentence provides an easy interface to generate structured
    sentences where each word is randomly generated.

    Example::

        >>> s = RandomSentence(nouns=["car", "cat", "mouse"], verbs=["eat"])
        >>> s2 = RandomSentence()

    :param nouns: a list of nouns that will be used to generate random nouns.
        Defaults to None.
    :type nouns: list, optional
    :param verbs: a list of verbs that will be used to generate random verbs.
        Defaults to None.
    :type verbs: list, optional
    :param adjectives: a list of adjectives that will be used to generate
        random adjectives. Defaults to None.
    :type adjectives: list, optional
    """

    def __init__(
        self,
        rng: random.Random = None,
        nouns: Optional[List[str]] = None,
        verbs: Optional[List[str]] = None,
        adjectives: Optional[List[str]] = None,
    ):
        noun = nouns or Defaults.NOUNS
        verb = verbs or Defaults.VERBS
        adjective = adjectives or Defaults.ADJECTIVES
        self._rng = rng or random.Random()
        self.gen = RandomWord(noun=noun, verb=verb, adjective=adjective, rng=self._rng)

    # Randomly generate bare bone sentences
    def bare_bone_sentence(self) -> str:
        """Generate a bare-bone sentence in the form of
        ``[(article)] [subject (noun)] [predicate (verb)].``. For example:
        ``The cat runs.``.

        Example::

            >>> s.bare_bone_sentence()

        :return: string in the form of a bare bone sentence where each word is
            randomly generated
        :rtype: str
        """
        the_noun = _with_article(self.gen.word(include_categories=["noun"]), self._rng)
        the_verb = _present_tense(self.gen.word(include_categories=["verb"]))

        return f"{the_noun.capitalize()} {the_verb}."

    def simple_sentence(self) -> str:
        """Generate a simple sentence in the form of
        ``[(article)] [subject (noun)] [predicate (verb)] [direct object (noun)].``.
        For example: ``The cake plays golf``.

        Example::

            >>> s.simple_sentence()

        :return: a string in the form of a simple sentence where each word is
            randomly generated
        :rtype: str
        """
        the_direct_object = self.gen.word(include_categories=["noun"])
        the_bare_bone_sentence = self.bare_bone_sentence()[:-1]

        return f"{the_bare_bone_sentence} {the_direct_object}."

    def bare_bone_with_adjective(self) -> str:
        """Generate a bare-bone sentence with an adjective in the form of:
        ``[(article)] [(adjective)] [subject (noun)] [predicate (verb)].``. For
        example: ``The skinny cat reads.``

        Example::

            >>> s.bare_bone_with_adjective()

        :return: a string in the form of a bare-bone sentence with an adjective
            where each word is randomly generated
        :rtype: str
        """
        the_noun = self.gen.word(include_categories=["noun"])
        the_verb = _present_tense(self.gen.word(include_categories=["verb"]))
        the_adjective = _with_article(self.gen.word(include_categories=["adjective"]), self._rng)

        return f"{the_adjective.capitalize()} {the_noun} {the_verb}."

    def sentence(self) -> str:
        """Generate a simple sentence with an adjective in the form of:
        ``[(article)] [(adjective)] [subject (noun)] [predicate (verb)]
        [direct object (noun)].``. For example:
        ``The green orange likes food.``

        Example::

            >>> s.sentence()

        :return: a string in the form of a simple sentence with an adjective
            where each word is randomly generated
        :rtype: str
        """
        the_bare_bone_with_adjective = self.bare_bone_with_adjective()[:-1]
        the_direct_object = self.gen.word(include_categories=["noun"])

        return f"{the_bare_bone_with_adjective} {the_direct_object}."

# Handles all TTS-related operations
class TTSHandler:
    
    def __init__(self, tts_base_path: str, sample_rates: List[int], log_level: str = "INFO", print_text: bool=False):
        self.print_text=print_text
        self.tts_base_path = tts_base_path
        self.sample_rates = sample_rates
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.log_level = log_level
        self.rng = random.Random()
        self.rs = RandomSentence(rng=self.rng)
        self.elevenlabs_client = None
        self.elevenlabs_voice_ids = []

    def _init_elevenlabs_client(self):
        self.elevenlabs_client = ElevenLabs(api_key=self.api_key)

    def _query_voice_ids(self):
        output = []

        self._init_elevenlabs_client()
        
        res = self.elevenlabs_client.voices.search()
        has_more = res.has_more
        next_page_token = res.next_page_token
        output += [v.voice_id for v in res.voices]
        while has_more:
            res = self.elevenlabs_client.voices.search(next_page_token=next_page_token)
            output += [v.voice_id for v in res.voices]
            has_more = res.has_more
            next_page_token = res.next_page_token
        return output
    
    def get_all_elevenlabs_voice_ids(self, max_tries=10):
        tries = 0
        while tries < max_tries:
            try: 
                self.elevenlabs_voice_ids = self._query_voice_ids()
                success = True
                subnet_logger(
                    severity="TRACE",
                    message=f"Queried ElevenLabs voice ids: {self.elevenlabs_voice_ids}. Length: {len(self.elevenlabs_voice_ids)}",
                    log_level=self.log_level
                )
                return

            except Exception as e:
                tries_remaining = max_tries - tries - 1
                subnet_logger(
                    severity="ERROR",
                    message=f"Error querying ElevenLabs voice ids: {e}. Tries remaining: {tries_remaining}",
                    log_level=self.log_level
                )
                tries += 1

    # Generates unique sentences for TTS 
    def _generate_random_sentence(self) -> str:
        output = ""
        for _ in range(0,4):
            choice = self.rng.randint(0,3)
            if choice == 0:
                output += self.rs.simple_sentence() + " "
            elif choice == 1:
                output += self.rs.bare_bone_with_adjective() + " "
            else:
                output += self.rs.sentence() + " "

        if self.print_text:
            print(output)
        return output
        
    # Generates one output TTS file at correct sample rate
    def _do_single_elevenlabs_tts_query(self, tts_file_path: str, sample_rate: int, voice: str = 'random'):
        # voice control
        if voice == 'random' or voice not in self.elevenlabs_voice_ids:
            voice = self.rng.choice(self.elevenlabs_voice_ids)
        
        # call with client 
        try:
            self._init_elevenlabs_client()
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=self._generate_random_sentence(),
                voice_id=voice,
                model_id="eleven_multilingual_v2",
                output_format="pcm_44100"
            )
            save(audio=audio, filename=tts_file_path)
        # raise error if it fails
        except Exception as e:
            subnet_logger(
                severity="ERROR",
                message=f"Could not get TTS audio file from ElevenLabs because of error: {e}",
                log_level=self.log_level
            )
        # resample in place if necessary
        try:
            # Load the generated TTS audio file
            audio_data, sr = librosa.load(tts_file_path, sr=None)
            
            # Check if sample rate matches
            if sr != sample_rate:
                # Resample the audio
                audio_data_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
                
                # Write the resampled audio back to the same file
                sf.write(tts_file_path, audio_data_resampled, sample_rate)
                
                # Log the resampling action
                subnet_logger(
                    severity="TRACE",
                    message=f"Resampled audio file '{tts_file_path}' from {sr} Hz to {sample_rate} Hz.",
                    log_level=self.log_level
                )
        except Exception as e:
            subnet_logger(
                severity="ERROR",
                message=f"Error during resampling of '{tts_file_path}': {e}",
                log_level=self.log_level
            )

    # Creates TTS dataset of length n at specified sample rate
    def create_elevenlabs_tts_dataset(self, sample_rate: int, n:int, for_miner: bool = False):
        # define output file location and make directory if it doesn't exist
        if for_miner: 
            output_dir = self.tts_base_path
        else:
            output_dir = os.path.join(self.tts_base_path, str(sample_rate))
        os.makedirs(output_dir, exist_ok=True)
        # count to n and make files
        for i in range(n):
            self._do_single_elevenlabs_tts_query(
                tts_file_path = os.path.join(output_dir, (str(i) + ".wav")),
                sample_rate=sample_rate
            )

    # Create TTS dataset of length n for all sample rates
    def create_elevenlabs_tts_dataset_for_all_sample_rates(self, n:int, seed:int = None):

        if seed:
            self.rng = random.Random(seed)

        self.rs = RandomSentence(
            rng=self.rng
        )
            
        for sample_rate in self.sample_rates: 
            self.create_elevenlabs_tts_dataset(
                sample_rate=sample_rate,
                n=n
            )