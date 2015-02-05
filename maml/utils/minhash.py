# -*- coding: utf-8 -*-
"""
Minhash.py

Created on Tue Nov 11 12:03:45 2014

@author: epyzerknapp
"""

import numpy as np
import numpy.random as npr

class Hasher(object):
    '''
    This is a transcoder class from string to minhash representation.
    '''
    def __init__(self, input_length=None, output_length=16, **kwargs):
        """
        Initialization.

        Keyword Arguments:

        input_length : required, int, the length of the string
        to be hashed.

        output_length : optional, int, default 16, the length of the
        output list

        """
        assert input_length is not None
        self.indices = [npr.permutation(input_length) for _ in
        xrange(0,output_length)]
        self.input_length = input_length
        self.output_length = output_length
        self.string = None
    def update(self, string):
        """
        Read in a string to digest.
        Nomeclature inspired by hashlib

        Arguments:
        string: required: str, the string to be hashed, must only
        contain the characters 0,1

        Example Usage:

        >>> s1 = '1010001010'
        >>> hasher = Hasher(input_length=len(s1), output_length=5)
        >>> hasher.update(s1)
        >>> print hasher.string
        '1010001010'

        """
        assert len(string) == self.input_length
        try:
            int(string,2)
        except ValueError:
            raise StandardError('Must be a binary string')
        self.string = string

    def digest(self):
        """
        Return the minhash of the input string as a list

        No Arguments

        Example Usage:
        >>> s1 = '1111111111'
        >>> hasher = Hasher(input_length=len(s1), output_length=5)
        >>> hasher.update(s1)
        >>> minhash = hasher.digest()
        >>> print minhash
        [1,1,1,1,1]

        """
        string = np.array(list(self.string))
        output = []
        for index in self.indices:
            first_non_zero = np.argmax(string[index])
            if first_non_zero == 0 and string[index][0] == '0':
                first_non_zero = self.input_length
            output.append(first_non_zero)
        return output




if __name__ == '__main__':
    minhash = Hasher(input_length=10,output_length=30)
    minhash.update('1001000000')
    print minhash.digest()
    minhash.update('0000000000')
    print minhash.digest()
    minhash.update('1111111111')
    print minhash.digest()

