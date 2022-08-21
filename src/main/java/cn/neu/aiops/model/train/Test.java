package cn.neu.aiops.model.train;

public class Test {
	public static void main(String[] args) {
		Cal cal = getService();

		int res = cal.add(1, 2);

		System.out.println(res);
	}

	public static Cal getService() {
		return new Service();
	}
}

interface Cal {
	int add(int a, int b);
}

class Service implements Cal {

	@Override
	public int add(int a, int b) {
		return a + b;
	}

}
