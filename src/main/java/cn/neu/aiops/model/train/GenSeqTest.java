package cn.neu.aiops.model.train;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

public class GenSeqTest {

	public static final String dirBase = Util.WORK_DIR + "test-seq\\";
	public static final String filePrefix = "seq-";

	private int[] normal = new int[] { 1, 2, 3 };

	private int[] signal = new int[] { 6, 7, 8 };

	private int exception = 10;

	public List<List<Writable>> main0() throws Exception {
		Random r = new Random();

		List<List<Writable>> data = new ArrayList<>();

		for (int i = 1; i <= 40; i++) {
			// 1-30
			// 31-39
			// 40;
			List<Writable> list = new ArrayList<Writable>();
			if (i < 30) {
				if (hasProbality(0.05)) {
					list.add(new DoubleWritable(exception));
					list.add(new IntWritable(0));
				} else {
					int idx = r.nextInt(3);
					list.add(new DoubleWritable(normal[idx]));
					list.add(new IntWritable(0));
				}
			} else if (i >= 30 && i <= 39) {
				int idx = r.nextInt(3);
				list.add(new DoubleWritable(signal[idx]));
				list.add(new IntWritable(0));
			} else {
				list.add(new DoubleWritable(exception));
				list.add(new IntWritable(1));
			}
			data.add(list);
		}

		int split = r.nextInt(data.size());
		List<List<Writable>> subList0 = data.subList(0, split);
		List<List<Writable>> subList1 = data.subList(split, data.size());

		List<List<Writable>> rtv = new ArrayList<>();
		rtv.addAll(subList1);
		rtv.addAll(subList0);

		return rtv;
	}

	public static void main(String[] args) throws Exception {
		GenSeqTest genSeqTest = new GenSeqTest();

		for (int i = 0; i < 100; i++) {
			List<List<Writable>> data = genSeqTest.main0();
			RecordWriter csvWriter = new CSVRecordWriter();
			csvWriter.initialize(new FileSplit(new File(dirBase + filePrefix + i + ".csv")),
					new NumberOfRecordsPartitioner());

			csvWriter.writeBatch(data);
			csvWriter.close();
		}
		
		System.out.println("Done!");
	}

	public static boolean hasProbality(double prob) {
		double r = Math.random();
		if (r < prob)
			return true;
		else
			return false;
	}

}
