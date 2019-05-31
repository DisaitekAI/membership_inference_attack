#! /usr/bin/env ruby

#  Usage:
#    ./process-logs.rb <log-file-path>
#    
#  Example:
#    ./process-logs.rb reports/Statistics_report_3/Logs

require 'nyaplot'
  
logPath = ARGV[0]
raise ArgumentError.new('the given path does not exists') if !File.exists?(logPath)
  
logText = IO.read(ARGV[0])

matches = logText.scan(/balanced_accuracy\n\[(0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+)\]/)
size = matches.size
print "#{size / 50} experiments found\n"

x = []
y = []

for i in 0..(size / 50) - 1 do
  
  expNumber = 0.0
  meanExpAccuracy = 0
  for j in 0..4 do
    meanMIAaccuracy = 0
    for k in 0..9 do
      index = i * 50 + j * 10 + k
      accTable = matches[index].map { |m| m.to_f }
      meanMIAaccuracy += accTable[-1]
    end
    meanMIAaccuracy /= 10.0 
    meanExpAccuracy += meanMIAaccuracy
    expNumber += 1.0
  end

  x << 1 + i*4
  if expNumber == 0
    y << 0.5
  else
    meanExpAccuracy /= expNumber
    y << meanExpAccuracy
  end
  
  print "mean MIA models accuracy for experiment #{i}: #{meanExpAccuracy}\n"
end

plot = Nyaplot::Plot.new
plot.x_label('shadow convolution filter number')
plot.y_label('mean accuracy of MIA attack models')

sc = plot.add(:scatter, x, y)
sc.color(Nyaplot::Colors.qual)

logDir = File.dirname(logPath)
plot.export_html(logDir + '/mean-model-accuracy.html')
