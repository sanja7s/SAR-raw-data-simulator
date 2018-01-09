def generateIdealChirp(chirpStartFreq,chirpStopFreq,chirpLength,rangeSamplingRate):
    pulseDuration=chirpLength/rangeSamplingRate
    k=(chirpStopFreq-chirpStartFreq)/pulseDuration
    t=np.linspace(0,pulseDuration,chirpLength)
    return np.asfortranarray(np.exp(1j*np.pi*(2*chirpStartFreq*t+k*t*t)).reshape(chirpLength,1)).astype(np.complex64)


def test():
	generateIdealChirp(30,90,10,75)