#include "Application/Application.hpp"

int main(int argc, char **argv)
{
    auto app = Application::GetApplication();
	app->Start(argc,argv);
    return 0;
}

// int PoolTestMain(int argc, char **argv)
// {
//     DECLARE_FUNCTOR(NoParamFunctor, void);
//     DECLARE_FUNCTOR_OneParam(FunctorOneParam, void, int);

//     RenderAPI::Thread::TThreadPool pool(3);

//     TTVector<int> s1 = {1, 2, 3};
//     int ans1 = 5;

//     TTVector<int> s2 = {4, 5};
//     int ans2 = 0;

//     TTVector<int> s3 = {8, 9, 10};
//     int ans3 = 0;

//     auto funcNoParam = TFunctorBase::Create(NoParamFunc);
//     auto funcOneParam = TFunctorBase::Create(OneParamFunc, ans1);

//     auto id1 = pool.AddTask(funcNoParam);
//     auto id2 = pool.AddTask(funcOneParam);
    
//     if(pool.IsDone(id1))
//     {
//         ///
//     }
//     else
//     {
//         pool.Wait(id1);
//     }
//     pool.WaitAll();

//     return 0;
// }
