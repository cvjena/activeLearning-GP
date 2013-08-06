function [tp, fp] = roc(gtLabels, scores)


    noInstances = size(scores,1);

    % sort by classifier output 
    try % compatibility with old Matlab versions
        [scores,idx] = sort(scores, 'descend');
        gtLabels       = gtLabels(idx) > 0;
    catch
        [scores,idx] = sort(full(scores), 'descend');
        gtLabels       = gtLabels(idx) > 0;    
    end

    % generate ROC

    noPositives     = sum( gtLabels );
    noNegatives     = noInstances - noPositives;

    if ( noPositives <= 0 )
        error('roc.m: no positive examples given!');
    end

    if ( noNegatives <= 0 )
        error('roc.m: no negative examples given!');
    end

    %init
    fp    = zeros( noInstances+2,1 );
    tp    = zeros( noInstances+2,1 );
    FP    = 0;
    TP    = 0;
    n     = 1;
    yprev = -realmax;

    % loop over scores
    for i=1:noInstances

        %did the score change?
       if ( scores(i) ~= yprev )
          tp(n) = TP/noPositives;
          fp(n) = FP/noNegatives; 
          yprev = scores(i);
          n     = n + 1;
       end

       % update TP and FP rate for the current score
       if ( gtLabels(i) == 1 )
          TP = TP + 1;
       else
          FP = FP + 1;
       end

    end

    %that's it :)
    tp(n) = 1;
    fp(n) = 1;
    fp    = fp(1:n);
    tp    = tp(1:n);

end
