function ranges = split_range(numPixels, numSubRegions, overlap)

lenRegion = sqrt(numPixels);
numSide = sqrt(numSubRegions);
if numSide ~= floor(numSide) || lenRegion ~= floor(lenRegion)
    error('numSubRegions and numPixels must be the square of an integer value');
end
if overlap
    if lenRegion == odd(lenRegion)
	    lenSubRegion = odd(2*lenRegion/(numSide+1),0);
	    lenSubRegion = max(3,lenSubRegion);
    else
        lenSubRegion = odd(2*lenRegion/(numSide+1))-1;
    end
else
    lenSubRegion = ceil(lenRegion/numSide);
end
hlen = (lenSubRegion-1)/2;
if numSide == 1
  spacing_stage = 1; 
else
  spacing_stage = (lenRegion-hlen-(1+hlen))/(numSide-1);
end
for r = 1:numSide
    cent = 1 + hlen + (r-1)*spacing_stage;
    ranges(r,:) = round([cent-hlen:cent+hlen]);
end
ranges
