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

matches = logText.scan(/\[(0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+), (0\.[0-9]+)\]/)
size = matches.size

x = []
y = []

for i in 0..(size / 10) - 1 do
  index = i * 10
  meanAccuracy = 0
  
  for j in index..(index + 9) do
    accTable = matches[j].map { |m| m.to_f }
    meanAccuracy += accTable.max
  end
  
  meanAccuracy /= 10.0
  
  x.append(i * 5)
  y.append(meanAccuracy)
  
  print "mean MIA models accuracy for experiment #{i}: #{meanAccuracy}\n"
end

plot = Nyaplot::Plot.new
plot.x_label('shadow number')
plot.y_label('mean accuracy')

sc = plot.add(:scatter, x, y)
sc.color(Nyaplot::Colors.qual)

logDir = File.dirname(logPath)
plot.export_html(logDir + '/mean-model-accuracy.html')
